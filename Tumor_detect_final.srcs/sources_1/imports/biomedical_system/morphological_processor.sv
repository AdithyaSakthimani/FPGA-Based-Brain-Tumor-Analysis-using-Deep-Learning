`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: Department of Computer Engineering, University of Peradeniya
// Engineer: Mahela Ekanayake, Chaminda Weerasinghe
//
// Module Name: morphological_processor.sv
// Description: Erosion and dilation over a 3x3 structuring element.
//
//   Erosion:  output = 0  if any neighbour in SE is 0, else 255
//             Equivalent to: output = min(neighbourhood)
//   Dilation: output = 255 if any neighbour in SE is 255, else 0
//             Equivalent to: output = max(neighbourhood)
//
//   Min/max extraction reuses the bitonic sort network from median_filter.sv.
//   Border pixels (outermost ring for 3x3 SE) are set to 0.
//
//   mode = 0 → erosion (min)
//   mode = 1 → dilation (max)
//
// Dependencies: none (bitonic sort inlined)
//////////////////////////////////////////////////////////////////////////////////

module morphological_processor #(
    parameter R_I = 7, C_I = 7, W_I = 8,
    parameter R_K = 3, C_K = 3
)(
    input  logic                        clk, cen,
    input  logic                        mode,          // 0=erosion, 1=dilation
    input  logic [R_I*C_I*W_I-1:0]     img_flat,
    output logic [R_I*C_I*W_I-1:0]     final_img_flat
);

    localparam VECTOR_SIZE = R_K * C_K;
    localparam LEVEL       = $clog2(VECTOR_SIZE);
    localparam DEPTH       = LEVEL * (LEVEL + 1) / 2;
    localparam LEAVES      = 2**LEVEL;

    // Unpack flat bus
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

    // Per-pixel bitonic sort → extract min (index 0) or max (index LEAVES-1)
    generate
        for (gi = 0; gi < R_I; gi++) begin : row
            for (gj = 0; gj < C_I; gj++) begin : col

                if (gi == 0 || gi == R_I-1 || gj == 0 || gj == C_I-1) begin : border
                    always_ff @(posedge clk) begin
                        if (cen) final_img[gi][gj] <= '0;
                    end
                end else begin : interior

                    // Bitonic sort tree (same structure as median_filter.sv)
                    logic [W_I-1:0] tree [0:DEPTH][0:LEAVES-1];

                    // Load leaves
                    genvar rk, ck;
                    for (rk = 0; rk < R_K; rk++) begin : ldr
                        for (ck = 0; ck < C_K; ck++) begin : ldc
                            always_ff @(posedge clk) begin
                                if (cen) begin
                                    automatic int ri = gi + rk - (R_K-1)/2;
                                    automatic int ci = gj + ck - (C_K-1)/2;
                                    if (ri >= 0 && ri < R_I && ci >= 0 && ci < C_I)
                                        tree[0][rk*C_K + ck] <= img[ri][ci];
                                    else
                                        tree[0][rk*C_K + ck] <= (mode == 0) ? 8'hFF : 8'h00;
                                end
                            end
                        end
                    end

                    // Pad remaining leaves
                    genvar pi;
                    for (pi = VECTOR_SIZE; pi < LEAVES; pi++) begin : pad
                        always_ff @(posedge clk) begin
                            if (cen)
                                tree[0][pi] <= (mode == 0) ? 8'hFF : 8'h00;
                        end
                    end

                    // Bitonic sort stages (ascending order → min at [0], max at [LEAVES-1])
                    genvar l, a, ii, jj, kk;
                    for (l = 0; l < LEVEL; l++) begin : lvl
                        for (a = 0; a < LEAVES/(2**(l+1)); a++) begin : grp
                            for (ii = 0; ii <= l; ii++) begin : stage
                                for (jj = 0; jj < 2**ii; jj++) begin : pair
                                    for (kk = 0; kk < (2**(l-ii)); kk++) begin : elem
                                        always_ff @(posedge clk) begin
                                            if (cen) begin
                                                automatic int idx0 = kk + jj*(2**(l-ii+1)) + a*(2**(l+1));
                                                automatic int idx1 = idx0 + (2**(l-ii));
                                                automatic int d    = l*(l+1)/2 + ii;
                                                // Always sort ascending (min at 0, max at LEAVES-1)
                                                if ($unsigned(tree[d][idx0]) > $unsigned(tree[d][idx1])) begin
                                                    tree[d+1][idx0] <= tree[d][idx1];
                                                    tree[d+1][idx1] <= tree[d][idx0];
                                                end else begin
                                                    tree[d+1][idx0] <= tree[d][idx0];
                                                    tree[d+1][idx1] <= tree[d][idx1];
                                                end
                                            end
                                        end
                                    end
                                end
                            end
                        end
                    end

                    // Extract result: erosion → min (index 0), dilation → max (index LEAVES-1)
                    // Then apply binary threshold: any non-zero → 255 for dilation,
                    // any non-255 → 0 for erosion (handles binary input {0,255})
                    always_comb begin
                        if (mode == 0)
                            // Erosion: min of neighbourhood
                            final_img[gi][gj] = (tree[DEPTH][0] == 8'd0) ? 8'd0 : 8'd255;
                        else
                            // Dilation: max of neighbourhood
                            final_img[gi][gj] = (tree[DEPTH][LEAVES-1] == 8'd255) ? 8'd255 : 8'd0;
                    end

                end
            end
        end
    endgenerate

endmodule
