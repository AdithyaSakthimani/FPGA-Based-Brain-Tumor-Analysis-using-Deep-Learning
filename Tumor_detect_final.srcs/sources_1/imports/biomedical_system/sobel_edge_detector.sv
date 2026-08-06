`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: Department of Computer Engineering, University of Peradeniya
// Engineer: Mahela Ekanayake, Chaminda Weerasinghe
//
// Module Name: sobel_edge_detector.sv
// Description: Parallel Sobel edge detection over a 3x3 kernel for every pixel
//              in the R_I x C_I image simultaneously.
//              Gx kernel: [[-1,0,1],[-2,0,2],[-1,0,1]]
//              Gy kernel: [[-1,-2,-1],[0,0,0],[1,2,1]]
//              G = min(|Gx| + |Gy|, 255)  (L1-norm, saturated to 8-bit)
//              Border pixels (outermost ring) are set to 0.
//
// Dependencies: none
//////////////////////////////////////////////////////////////////////////////////

module sobel_edge_detector #(
    parameter R_I = 7, C_I = 7, W_I = 8,
    parameter R_K = 3, C_K = 3
)(
    input  logic                        clk, cen,
    input  logic [R_I*C_I*W_I-1:0]     img_flat,
    output logic [R_I*C_I*W_I-1:0]     final_img_flat
);

    // Unpack flat bus to 2-D pixel array
    logic [W_I-1:0] img      [0:R_I-1][0:C_I-1];
    logic [W_I-1:0] final_img[0:R_I-1][0:C_I-1];

    genvar gi, gj;
    generate
        for (gi = 0; gi < R_I; gi++) begin : unpack_r
            for (gj = 0; gj < C_I; gj++) begin : unpack_c
                assign img[gi][gj] = img_flat[(gi*C_I + gj)*W_I +: W_I];
                assign final_img_flat[(gi*C_I + gj)*W_I +: W_I] = final_img[gi][gj];
            end
        end
    endgenerate

    // Sobel kernels (signed weights)
    // Kx: col weights  -1  0  +1  (row weights 1 2 1)
    // Ky: row weights  -1  0  +1  (col weights 1 2 1)
    localparam signed [2:0] KX [0:2][0:2] = '{
        '{-1, 0, 1},
        '{-2, 0, 2},
        '{-1, 0, 1}
    };
    localparam signed [2:0] KY [0:2][0:2] = '{
        '{-1, -2, -1},
        '{ 0,  0,  0},
        '{ 1,  2,  1}
    };

    // Per-pixel computation
    generate
        for (gi = 0; gi < R_I; gi++) begin : row
            for (gj = 0; gj < C_I; gj++) begin : col

                // Border pixels → 0
                if (gi == 0 || gi == R_I-1 || gj == 0 || gj == C_I-1) begin : border
                    always_ff @(posedge clk) begin
                        if (cen) final_img[gi][gj] <= '0;
                    end
                end else begin : interior

                    // Accumulate Gx and Gy with signed arithmetic
                    // W_I+4 bits is sufficient: max |Gx| = 4*255 = 1020 < 2^10
                    logic signed [W_I+3:0] gx_acc, gy_acc;
                    logic signed [W_I+3:0] abs_gx, abs_gy;
                    logic        [W_I+3:0] l1;

                    always_ff @(posedge clk) begin
                        if (cen) begin
                            gx_acc = '0;
                            gy_acc = '0;
                            for (int rk = 0; rk < R_K; rk++) begin
                                for (int ck = 0; ck < C_K; ck++) begin
                                    automatic int ri = gi + rk - (R_K-1)/2;
                                    automatic int ci = gj + ck - (C_K-1)/2;
                                    if (ri >= 0 && ri < R_I && ci >= 0 && ci < C_I) begin
                                        gx_acc = gx_acc + KX[rk][ck] * $signed({1'b0, img[ri][ci]});
                                        gy_acc = gy_acc + KY[rk][ck] * $signed({1'b0, img[ri][ci]});
                                    end
                                end
                            end
                            abs_gx = (gx_acc[W_I+3]) ? -gx_acc : gx_acc;
                            abs_gy = (gy_acc[W_I+3]) ? -gy_acc : gy_acc;
                            l1     = abs_gx + abs_gy;
                            // Saturate to 8-bit
                            final_img[gi][gj] <= (l1 > 255) ? 8'd255 : l1[W_I-1:0];
                        end
                    end
                end
            end
        end
    endgenerate

endmodule
