#include <iostream>
#include <fstream>
#include <vector>
#include <cstdint>
#include <chrono>

int main(int argc, char* argv[]) {
    if (argc < 3) {
        std::cerr << "Usage: ./zbf16_packer <input.gguf> <output.zbf16>\n";
        return 1;
    }
    std::string input_path = argv[1];
    std::string output_path = argv[2];

    std::ifstream in(input_path, std::ios::binary | std::ios::ate);
    if (!in) {
        std::cerr << "Error: Cannot open input file: " << input_path << "\n";
        return 1;
    }
    size_t total_size = in.tellg();
    in.seekg(0, std::ios::beg);

    std::ofstream out(output_path, std::ios::binary);
    const uint32_t MAGIC = 0x3146425A; // "ZBF1"
    const uint8_t PIVOT = 0x79;
    out.write(reinterpret_cast<const char*>(&MAGIC), 4);
    out.write(reinterpret_cast<const char*>(&total_size), sizeof(total_size));
    out.write(reinterpret_cast<const char*>(&PIVOT), 1);

    const size_t CHUNK_ELEMENTS = 8388608;
    std::vector<uint16_t> buffer(CHUNK_ELEMENTS);
    std::vector<uint8_t> compressed_upper;
    std::vector<uint8_t> raw_lower;
    compressed_upper.reserve(CHUNK_ELEMENTS);
    raw_lower.reserve(CHUNK_ELEMENTS);

    auto start = std::chrono::high_resolution_clock::now();
    std::cout << "Compressing to .zbf16 format...\n";

    while (in.read(reinterpret_cast<char*>(buffer.data()), CHUNK_ELEMENTS * 2) || in.gcount() > 0) {
        size_t bytes_read = in.gcount();
        size_t count = bytes_read / 2;
        if (count == 0) break;

        compressed_upper.clear();
        raw_lower.clear();

        uint64_t bit_buf = 0;
        int bit_cnt = 0;

        for (size_t i = 0; i < count; i++) {
            uint16_t val = buffer[i];
            uint8_t upper = (val >> 8) & 0xFF;
            raw_lower.push_back(val & 0xFF);

            uint8_t rotated = ((upper & 0x7F) << 1) | ((upper >> 7) & 1);
            uint8_t diff = rotated ^ PIVOT;

            for (int shift = 6; shift >= 0; shift -= 2) {
                uint8_t pair = (diff >> shift) & 0b11;
                uint16_t code = (pair == 0) ? 0b0 : (pair == 1) ? 0b10 : (pair == 2) ? 0b110 : 0b111;
                int len = (pair == 0) ? 1 : (pair == 1) ? 2 : 3;

                bit_buf = (bit_buf << len) | code;
                bit_cnt += len;

                while (bit_cnt >= 8) {
                    bit_cnt -= 8;
                    compressed_upper.push_back((bit_buf >> bit_cnt) & 0xFF);
                }
            }
        }
        if (bit_cnt > 0) {
            compressed_upper.push_back((bit_buf << (8 - bit_cnt)) & 0xFF);
        }

        uint32_t c_cnt = static_cast<uint32_t>(count);
        uint32_t u_len = static_cast<uint32_t>(compressed_upper.size());
        out.write(reinterpret_cast<const char*>(&c_cnt), 4);
        out.write(reinterpret_cast<const char*>(&u_len), 4);
        out.write(reinterpret_cast<const char*>(compressed_upper.data()), u_len);
        out.write(reinterpret_cast<const char*>(raw_lower.data()), c_cnt);
    }
    out.close();

    auto end = std::chrono::high_resolution_clock::now();
    double elapsed = std::chrono::duration<double>(end - start).count();
    std::ifstream check(output_path, std::ios::binary | std::ios::ate);
    size_t new_size = check.tellg();

    std::cout << "Original Size   : " << (total_size / (1024.0 * 1024.0 * 1024.0)) << " GB\n";
    std::cout << "Compressed Size : " << (new_size / (1024.0 * 1024.0 * 1024.0)) << " GB\n";
    std::cout << "Net Saved       : " << ((total_size - new_size) / (1024.0 * 1024.0)) << " MB\n";
    std::cout << "Time Elapsed    : " << elapsed << " s\n";
    return 0;
}
