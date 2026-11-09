#include <iostream>
#include <fstream>
#include <vector>
#include <cstdint>
#include <chrono>

int main(int argc, char* argv[]) {
    if (argc < 3) {
        std::cerr << "Usage: ./zbf16_unpacker <input.zbf16> <output.gguf>\n";
        return 1;
    }
    std::string input_path = argv[1];
    std::string output_path = argv[2];

    std::ifstream in(input_path, std::ios::binary);
    if (!in) {
        std::cerr << "Error: Cannot open .zbf16 file.\n";
        return 1;
    }

    uint32_t magic = 0;
    size_t orig_size = 0;
    uint8_t PIVOT = 0;

    in.read(reinterpret_cast<char*>(&magic), 4);
    in.read(reinterpret_cast<char*>(&orig_size), sizeof(orig_size));
    in.read(reinterpret_cast<char*>(&PIVOT), 1);

    std::ofstream out(output_path, std::ios::binary);
    std::vector<uint8_t> comp_upper;
    std::vector<uint8_t> raw_lower;
    std::vector<uint16_t> out_buf;

    auto start = std::chrono::high_resolution_clock::now();
    std::cout << "Decompressing .zbf16 to bit-exact GGUF...\n";

    while (in.peek() != EOF) {
        uint32_t s_count = 0, u_len = 0;
        if (!in.read(reinterpret_cast<char*>(&s_count), 4)) break;
        in.read(reinterpret_cast<char*>(&u_len), 4);

        comp_upper.resize(u_len);
        raw_lower.resize(s_count);
        out_buf.resize(s_count);

        in.read(reinterpret_cast<char*>(comp_upper.data()), u_len);
        in.read(reinterpret_cast<char*>(raw_lower.data()), s_count);

        size_t byte_idx = 0;
        int bit_idx = 7;

        auto read_bit = [&]() -> int {
            if (byte_idx >= u_len) return 0;
            int bit = (comp_upper[byte_idx] >> bit_idx) & 1;
            bit_idx--;
            if (bit_idx < 0) {
                bit_idx = 7;
                byte_idx++;
            }
            return bit;
        };

        for (uint32_t i = 0; i < s_count; i++) {
            uint8_t diff = 0;
            for (int p = 0; p < 4; p++) {
                int b1 = read_bit();
                uint8_t pair = 0;
                if (b1 == 0) {
                    pair = 0b00;
                } else {
                    int b2 = read_bit();
                    if (b2 == 0) {
                        pair = 0b01;
                    } else {
                        int b3 = read_bit();
                        pair = (b3 == 0) ? 0b10 : 0b11;
                    }
                }
                diff = (diff << 2) | pair;
            }
            uint8_t rotated = diff ^ PIVOT;
            uint8_t upper = ((rotated & 1) << 7) | (rotated >> 1);
            out_buf[i] = (static_cast<uint16_t>(upper) << 8) | raw_lower[i];
        }
        out.write(reinterpret_cast<const char*>(out_buf.data()), s_count * 2);
    }

    out.close();
    auto end = std::chrono::high_resolution_clock::now();
    double elapsed = std::chrono::duration<double>(end - start).count();
    std::cout << "Decompression complete! Time: " << elapsed << " s\n";
    return 0;
}
