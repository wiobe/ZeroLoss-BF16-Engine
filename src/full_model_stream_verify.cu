#include <iostream>
#include <fstream>
#include <vector>
#include <cstdint>
#include <chrono>
#include <cuda_runtime.h>

__global__ void gpu_bit_ac_kernel(
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

    uint32_t byte_pos = bit_imleci >> 3;
    int bit_offset = bit_imleci & 7;
    
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
            if (kalan_bit < 8) {
                cache |= (static_cast<uint32_t>(sikistirilmis_veri[byte_pos++]) << (24 - kalan_bit));
                kalan_bit += 8;
            }
            uint8_t ikili = 0;
            if ((cache & 0x80000000) == 0) {
                ikili = 0b00; cache <<= 1; kalan_bit -= 1;
            } else if ((cache & 0x40000000) == 0) {
                ikili = 0b01; cache <<= 2; kalan_bit -= 2;
            } else {
                ikili = ((cache & 0x20000000) == 0) ? 0b10 : 0b11;
                cache <<= 3; kalan_bit -= 3;
            }
            fark = (fark << 2) | ikili;
        }
        uint8_t duzenli = fark ^ merkez_deger;
        acilan_ust_byteler[cikti_baslangic + i] = ((duzenli & 1) << 7) | (duzenli >> 1);
    }
}

int main(int argc, char* argv[]) {
    std::string dosya_adi = (argc > 1) ? argv[1] : "Qwen_Qwen2.5-VL-7B-Instruct-bf16.gguf";
    std::ifstream dosya(dosya_adi, std::ios::binary | std::ios::ate);
    if (!dosya) {
        std::cerr << "Error: Cannot open " << dosya_adi << "!\n";
        return 1;
    }

    size_t toplam_dosya_boyutu = dosya.tellg();
    dosya.seekg(0, std::ios::beg);

    const size_t PARCA_SAYI_ADEDI = 8388608; // 16 MB bloklar
    const size_t PARCA_BYTE_BOYUTU = PARCA_SAYI_ADEDI * sizeof(uint16_t);
    const int BLOK_BOYUTU = 64;
    const int TOPLAM_BLOK = PARCA_SAYI_ADEDI / BLOK_BOYUTU;
    const uint8_t MERKEZ = 0x79;

    std::vector<uint16_t> okuma_tamponu(PARCA_SAYI_ADEDI);
    
    uint8_t *h_ust_byteler, *h_sikistirilmis, *h_dogrulama;
    uint32_t *h_blok_adresleri;
    cudaHostAlloc(&h_ust_byteler, PARCA_SAYI_ADEDI, cudaHostAllocDefault);
    cudaHostAlloc(&h_sikistirilmis, PARCA_SAYI_ADEDI * 2, cudaHostAllocDefault);
    cudaHostAlloc(&h_dogrulama, PARCA_SAYI_ADEDI, cudaHostAllocDefault);
    cudaHostAlloc(&h_blok_adresleri, TOPLAM_BLOK * sizeof(uint32_t), cudaHostAllocDefault);

    uint8_t *d_sikistirilmis, *d_acilan;
    uint32_t *d_blok_adresleri;
    cudaMalloc(&d_sikistirilmis, PARCA_SAYI_ADEDI * 2);
    cudaMalloc(&d_blok_adresleri, TOPLAM_BLOK * sizeof(uint32_t));
    cudaMalloc(&d_acilan, PARCA_SAYI_ADEDI);

    size_t toplam_ust_byte = 0;
    size_t toplam_sikistirilmis_byte = 0;
    size_t toplam_hata = 0;
    int parca_no = 0;

    std::cout << "Streaming Full 15 GB Model (Zero-Loss Verification Mode): " << dosya_adi << "\n";
    auto genel_baslangic = std::chrono::high_resolution_clock::now();

    while (dosya.read(reinterpret_cast<char*>(okuma_tamponu.data()), PARCA_BYTE_BOYUTU) || dosya.gcount() > 0) {
        size_t okunan_byte = dosya.gcount();
        size_t okunan_sayi = okunan_byte / sizeof(uint16_t);
        int aktif_blok_sayisi = okunan_sayi / BLOK_BOYUTU;
        if (aktif_blok_sayisi == 0) break;

        size_t islenen_sayi = aktif_blok_sayisi * BLOK_BOYUTU;

        uint64_t bit_tamponu = 0;
        int tampondaki_bit = 0;
        uint32_t toplam_bit = 0;
        size_t cikti_idx = 0;

        for (int b = 0; b < aktif_blok_sayisi; b++) {
            h_blok_adresleri[b] = toplam_bit;
            for (int i = 0; i < BLOK_BOYUTU; i++) {
                size_t idx = b * BLOK_BOYUTU + i;
                uint8_t ust = (okuma_tamponu[idx] >> 8) & 0xFF;
                h_ust_byteler[idx] = ust;

                uint8_t duzenli = ((ust & 0x7F) << 1) | ((ust >> 7) & 1);
                uint8_t fark = duzenli ^ MERKEZ;

                for (int kaydir = 6; kaydir >= 0; kaydir -= 2) {
                    uint8_t ikili = (fark >> kaydir) & 0b11;
                    uint16_t kod = (ikili == 0) ? 0b0 : (ikili == 1) ? 0b10 : (ikili == 2) ? 0b110 : 0b111;
                    int uzunluk = (ikili == 0) ? 1 : (ikili == 1) ? 2 : 3;

                    bit_tamponu = (bit_tamponu << uzunluk) | kod;
                    tampondaki_bit += uzunluk;
                    toplam_bit += uzunluk;

                    while (tampondaki_bit >= 8) {
                        tampondaki_bit -= 8;
                        h_sikistirilmis[cikti_idx++] = (bit_tamponu >> tampondaki_bit) & 0xFF;
                    }
                }
            }
        }

        // KRİTİK DÜZELTME: Parça sonundaki kalan yarım bitleri tampona boşalt (Flush)
        if (tampondaki_bit > 0) {
            h_sikistirilmis[cikti_idx++] = (bit_tamponu << (8 - tampondaki_bit)) & 0xFF;
            tampondaki_bit = 0;
            bit_tamponu = 0;
        }
        // GPU 32-bit önbellek okuması için güvenli son tampon (Padding)
        for (int p = 0; p < 8; p++) h_sikistirilmis[cikti_idx++] = 0;

        cudaMemcpy(d_sikistirilmis, h_sikistirilmis, cikti_idx, cudaMemcpyHostToDevice);
        cudaMemcpy(d_blok_adresleri, h_blok_adresleri, aktif_blok_sayisi * sizeof(uint32_t), cudaMemcpyHostToDevice);

        int threads = 256;
        int blocks = (aktif_blok_sayisi + threads - 1) / threads;
        gpu_bit_ac_kernel<<<blocks, threads>>>(d_sikistirilmis, d_blok_adresleri, d_acilan, aktif_blok_sayisi, BLOK_BOYUTU, MERKEZ);
        cudaDeviceSynchronize();

        cudaMemcpy(h_dogrulama, d_acilan, islenen_sayi, cudaMemcpyDeviceToHost);
        for (size_t i = 0; i < islenen_sayi; i++) {
            if (h_dogrulama[i] != h_ust_byteler[i]) toplam_hata++;
        }

        toplam_ust_byte += islenen_sayi;
        toplam_sikistirilmis_byte += cikti_idx;
        parca_no++;

        if (parca_no % 100 == 0) {
            std::cout << "Ilerleme: " << (parca_no * 16) << " MB islendi...\n";
        }
    }

    auto genel_bitis = std::chrono::high_resolution_clock::now();
    double toplam_sure_sn = std::chrono::duration<double>(genel_bitis - genel_baslangic).count();
    double kazanc_yuzdesi = (1.0 - (static_cast<double>(toplam_sikistirilmis_byte) / toplam_ust_byte)) * 100.0;

    std::cout << "\n=== FULL 15 GB QWEN MODEL STREAMING REPORT ===\n";
    std::cout << "Processed Chunks (16MB each) : " << parca_no << " blok\n";
    std::cout << "Original Upper-Plane Volume  : " << (toplam_ust_byte / (1024.0 * 1024.0)) << " MB\n";
    std::cout << "Compressed Upper-Plane Volume    : " << (toplam_sikistirilmis_byte / (1024.0 * 1024.0)) << " MB\n";
    std::cout << "Net Lossless Compression Gain        : %" << kazanc_yuzdesi << " (Tamami Kayipsiz)\n";
    std::cout << "Total End-to-End Elapsed Time       : " << toplam_sure_sn << " saniye\n";
    std::cout << "Total Bit Verification Errors           : " << toplam_hata << " (0 ise %100 Kusursuz!)\n";

    return 0;
}
