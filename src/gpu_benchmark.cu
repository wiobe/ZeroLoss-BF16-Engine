#include <iostream>
#include <fstream>
#include <vector>
#include <cstdint>
#include <cuda_runtime.h>

// OPTİMİZE EDİLMİŞ GPU ÇEKİRDEĞİ (32-bit Register Bit-Buffer + Tam Paralellik)
__global__ void gpu_bit_ac_hizli_kernel(
    const uint8_t* __restrict__ sikistirilmis_veri,
    const uint32_t* __restrict__ blok_baslangic_bitleri,
    uint8_t* __restrict__ acilan_ust_byteler,
    int toplam_blok,
    int blok_boyutu,
    uint8_t merkez_deger
) {
    int blok_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (blok_id >= toplam_blok) return;

    uint32_t bit_imleci = blok_baslangic_bitleri[blok_id];
    int cikti_baslangic = blok_id * blok_boyutu;

    // VRAM erişimini azaltmak için 32-bitlik yerel bit tamponu (Register)
    uint32_t byte_pos = bit_imleci >> 3;
    int bit_offset = bit_imleci & 7;
    
    // İlk 32 biti tek seferde çek
    uint32_t cache = (static_cast<uint32_t>(sikistirilmis_veri[byte_pos]) << 24) |
                     (static_cast<uint32_t>(sikistirilmis_veri[byte_pos + 1]) << 16) |
                     (static_cast<uint32_t>(sikistirilmis_veri[byte_pos + 2]) << 8) |
                     static_cast<uint32_t>(sikistirilmis_veri[byte_pos + 3]);
    cache <<= bit_offset;
    int kalan_bit = 32 - bit_offset;
    byte_pos += 4;

    for (int i = 0; i < blok_boyutu; i++) {
        uint8_t fark = 0;

        #pragma unroll
        for (int parca = 0; parca < 4; parca++) {
            // Tamponda 8 bitten az kaldıysa VRAM'den yeni byte doldur
            if (kalan_bit < 8) {
                cache |= (static_cast<uint32_t>(sikistirilmis_veri[byte_pos++]) << (24 - kalan_bit));
                kalan_bit += 8;
            }

            uint8_t ikili = 0;
            if ((cache & 0x80000000) == 0) {
                // '0' -> 00
                ikili = 0b00;
                cache <<= 1;
                kalan_bit -= 1;
            } else if ((cache & 0x40000000) == 0) {
                // '10' -> 01
                ikili = 0b01;
                cache <<= 2;
                kalan_bit -= 2;
            } else {
                // '110' -> 10, '111' -> 11
                ikili = ((cache & 0x20000000) == 0) ? 0b10 : 0b11;
                cache <<= 3;
                kalan_bit -= 3;
            }
            fark = (fark << 2) | ikili;
        }

        uint8_t duzenli = fark ^ merkez_deger;
        acilan_ust_byteler[cikti_baslangic + i] = ((duzenli & 1) << 7) | (duzenli >> 1);
    }
}

int main() {
    // Daha büyük veri setiyle gerçek PCIe bant genişliğini ölçelim (8 MB = 4.194.304 sayı)
    // Elimizdeki 1 MB veriyi 8 kez çoğaltarak 8 MB'lık gerçek bir Qwen katmanı simüle edelim
    const int TEK_KESIT = 524288;
    const int CARPAN = 8;
    const int TOPLAM_SAYI = TEK_KESIT * CARPAN; // 4.194.304 sayı (4 MB üst düzlem)
    const int BLOK_BOYUTU = 64; // 65.536 paralel GPU thread'i uyanacak!
    const int TOPLAM_BLOK = TOPLAM_SAYI / BLOK_BOYUTU;
    const uint8_t MERKEZ = 0x79;

    std::ifstream dosya("gercek_bf16_kesit.bin", std::ios::binary);
    std::vector<uint16_t> ham_kesit(TEK_KESIT);
    dosya.read(reinterpret_cast<char*>(ham_kesit.data()), TEK_KESIT * sizeof(uint16_t));

    // Pinned Memory (NVIDIA DMA motorunun doğrudan erişebildiği yüksek hızlı RAM)
    uint8_t *h_orijinal_ust, *h_sikistirilmis;
    uint32_t *h_blok_adresleri;
    cudaHostAlloc(&h_orijinal_ust, TOPLAM_SAYI, cudaHostAllocDefault);
    cudaHostAlloc(&h_sikistirilmis, TOPLAM_SAYI + 1024, cudaHostAllocDefault);
    cudaHostAlloc(&h_blok_adresleri, TOPLAM_BLOK * sizeof(uint32_t), cudaHostAllocDefault);

    uint64_t bit_tamponu = 0;
    int tampondaki_bit = 0;
    uint32_t toplam_bit_sayaci = 0;
    size_t cikti_byte_idx = 0;

    for (int b = 0; b < TOPLAM_BLOK; b++) {
        h_blok_adresleri[b] = toplam_bit_sayaci;
        for (int i = 0; i < BLOK_BOYUTU; i++) {
            int idx = (b * BLOK_BOYUTU + i) % TEK_KESIT;
            uint8_t ust = (ham_kesit[idx] >> 8) & 0xFF;
            h_orijinal_ust[b * BLOK_BOYUTU + i] = ust;

            uint8_t duzenli = ((ust & 0x7F) << 1) | ((ust >> 7) & 1);
            uint8_t fark = duzenli ^ MERKEZ;

            for (int kaydir = 6; kaydir >= 0; kaydir -= 2) {
                uint8_t ikili = (fark >> kaydir) & 0b11;
                uint16_t kod = (ikili == 0) ? 0b0 : (ikili == 1) ? 0b10 : (ikili == 2) ? 0b110 : 0b111;
                int uzunluk = (ikili == 0) ? 1 : (ikili == 1) ? 2 : 3;

                bit_tamponu = (bit_tamponu << uzunluk) | kod;
                tampondaki_bit += uzunluk;
                toplam_bit_sayaci += uzunluk;

                while (tampondaki_bit >= 8) {
                    tampondaki_bit -= 8;
                    h_sikistirilmis[cikti_byte_idx++] = (bit_tamponu >> tampondaki_bit) & 0xFF;
                }
            }
        }
    }
    // Güvenli taşma tamponu (Padding)
    for (int p = 0; p < 8; p++) h_sikistirilmis[cikti_byte_idx++] = 0;

    uint8_t *d_sikistirilmis, *d_acilan, *d_normal;
    uint32_t *d_blok_adresleri;
    cudaMalloc(&d_sikistirilmis, cikti_byte_idx);
    cudaMalloc(&d_blok_adresleri, TOPLAM_BLOK * sizeof(uint32_t));
    cudaMalloc(&d_acilan, TOPLAM_SAYI);
    cudaMalloc(&d_normal, TOPLAM_SAYI);

    // GPU'yu ısıt (Warmup)
    cudaMemcpy(d_normal, h_orijinal_ust, TOPLAM_SAYI, cudaMemcpyHostToDevice);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // 1. TEST: Normal Aktarım
    cudaEventRecord(start);
    cudaMemcpy(d_normal, h_orijinal_ust, TOPLAM_SAYI, cudaMemcpyHostToDevice);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float normal_ms = 0;
    cudaEventElapsedTime(&normal_ms, start, stop);

    // 2. TEST: Optimize Edilmiş Sıkıştırılmış Aktarım + GPU Açma
    cudaEventRecord(start);
    cudaMemcpy(d_sikistirilmis, h_sikistirilmis, cikti_byte_idx, cudaMemcpyHostToDevice);
    cudaMemcpy(d_blok_adresleri, h_blok_adresleri, TOPLAM_BLOK * sizeof(uint32_t), cudaMemcpyHostToDevice);
    
    int threadsPerBlock = 256;
    int blocksPerGrid = (TOPLAM_BLOK + threadsPerBlock - 1) / threadsPerBlock;
    gpu_bit_ac_hizli_kernel<<<blocksPerGrid, threadsPerBlock>>>(
        d_sikistirilmis, d_blok_adresleri, d_acilan, TOPLAM_BLOK, BLOK_BOYUTU, MERKEZ
    );
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float ozel_ms = 0;
    cudaEventElapsedTime(&ozel_ms, start, stop);

    // Doğrulama
    std::vector<uint8_t> dogrulama(TOPLAM_SAYI);
    cudaMemcpy(dogrulama.data(), d_acilan, TOPLAM_SAYI, cudaMemcpyDeviceToHost);
    int hata = 0;
    for (int i = 0; i < TOPLAM_SAYI; i++) if (dogrulama[i] != h_orijinal_ust[i]) hata++;

    std::cout << "=== RTX 3090 OPTIMIZED HARDWARE BENCHMARK (4 MB TENSOR) ===\n";
    std::cout << "Original Payload Size        : " << TOPLAM_SAYI << " Byte\n";
    std::cout << "Compressed Payload Size  : " << cikti_byte_idx << " Byte\n";
    std::cout << "Standard PCIe Transfer Time  : " << normal_ms << " ms\n";
    std::cout << "ZBF16 Engine (PCIe + GPU Kernel) : " << ozel_ms << " ms\n";
    std::cout << "Verification Bit Errors            : " << hata << "\n";

    return 0;
}
