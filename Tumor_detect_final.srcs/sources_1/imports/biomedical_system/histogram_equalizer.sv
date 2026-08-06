`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: Department of Computer Engineering, University of Peradeniya
// Engineer: Mahela Ekanayake, Chaminda Weerasinghe
//
// Module Name: histogram_equalizer.sv
// Description: Two-pass histogram equalisation for a single image tile.
//
//   Pass 1 (HIST):  Accumulate 256-bin histogram from all R_I*C_I pixels.
//   Pass 2 (CDF):   Compute CDF, find CDF_min, build 256-entry equalisation LUT.
//   Pass 3 (MAP):   Re-read input pixels and apply LUT to produce output.
//
//   Formula: out = round((CDF(in) - CDF_min) / (N_pixels - CDF_min) * 255)
//
//   The module operates on a flat pixel bus (R_I*C_I*W_I bits).
//   It accepts one tile per transaction (s_valid/s_ready handshake) and
//   emits the equalised tile on the master port.
//
//   Block RAM usage:
//     - hist_mem[256]  : 32-bit histogram bins  (M9K inferred)
//     - lut_mem[256]   : 8-bit equalisation LUT (M9K inferred)
//
// Dependencies: none
//////////////////////////////////////////////////////////////////////////////////

module histogram_equalizer #(
    parameter R_I = 7, C_I = 7, W_I = 8
)(
    input  logic                        clk, rstn,
    input  logic [R_I*C_I*W_I-1:0]     s_data,
    input  logic                        s_valid,
    output logic                        s_ready,
    output logic [R_I*C_I*W_I-1:0]     m_data,
    output logic                        m_valid,
    input  logic                        m_ready
);

    localparam NPIX    = R_I * C_I;
    localparam NBINS   = 256;
    localparam CNTW    = $clog2(NPIX) + 1;   // enough bits for pixel count

    // -----------------------------------------------------------------------
    // Block RAM: histogram and LUT
    // -----------------------------------------------------------------------
    logic [31:0]    hist_mem [0:NBINS-1];   // histogram bins
    logic [W_I-1:0] lut_mem  [0:NBINS-1];  // equalisation LUT

    // -----------------------------------------------------------------------
    // Pixel array unpacked from flat bus (registered on capture)
    // -----------------------------------------------------------------------
    logic [W_I-1:0] pix_buf [0:NPIX-1];
    logic [W_I-1:0] pix_out [0:NPIX-1];

    // -----------------------------------------------------------------------
    // FSM
    // -----------------------------------------------------------------------
    typedef enum logic [2:0] {
        IDLE,       // waiting for s_valid
        HIST,       // accumulate histogram (NPIX cycles)
        CDF,        // compute CDF + LUT (256 cycles)
        MAP,        // apply LUT to pixels (NPIX cycles)
        EMIT        // hold m_valid until m_ready
    } state_t;

    state_t state;

    logic [$clog2(NPIX)-1:0]  pix_cnt;
    logic [7:0]               bin_cnt;   // 0..255 for CDF pass
    logic [31:0]              cdf_acc;
    logic [31:0]              cdf_min;
    logic                     cdf_min_found;

    // -----------------------------------------------------------------------
    // FSM + datapath
    // -----------------------------------------------------------------------
    always_ff @(posedge clk or negedge rstn) begin
        if (!rstn) begin
            state         <= IDLE;
            s_ready       <= 1'b1;
            m_valid       <= 1'b0;
            pix_cnt       <= '0;
            bin_cnt       <= '0;
            cdf_acc       <= '0;
            cdf_min       <= '0;
            cdf_min_found <= 1'b0;
            // Clear histogram
            for (int i = 0; i < NBINS; i++) hist_mem[i] <= '0;
        end else begin
            case (state)

                // ----------------------------------------------------------
                IDLE: begin
                    m_valid <= 1'b0;
                    if (s_valid && s_ready) begin
                        // Capture pixel bus
                        for (int i = 0; i < NPIX; i++)
                            pix_buf[i] <= s_data[i*W_I +: W_I];
                        // Clear histogram
                        for (int i = 0; i < NBINS; i++) hist_mem[i] <= '0;
                        s_ready       <= 1'b0;
                        pix_cnt       <= '0;
                        cdf_acc       <= '0;
                        cdf_min_found <= 1'b0;
                        state         <= HIST;
                    end
                end

                // ----------------------------------------------------------
                // Accumulate histogram: one pixel per cycle
                HIST: begin
                    hist_mem[pix_buf[pix_cnt]] <= hist_mem[pix_buf[pix_cnt]] + 1;
                    if (pix_cnt == NPIX-1) begin
                        pix_cnt <= '0;
                        bin_cnt <= '0;
                        state   <= CDF;
                    end else begin
                        pix_cnt <= pix_cnt + 1;
                    end
                end

                // ----------------------------------------------------------
                // CDF pass: walk bins 0..255, accumulate CDF, build LUT
                CDF: begin
                    automatic logic [31:0] new_cdf;
                    automatic logic [31:0] lut_val;
                    new_cdf = cdf_acc + hist_mem[bin_cnt];
                    cdf_acc <= new_cdf;

                    // Track CDF_min (first non-zero bin)
                    if (!cdf_min_found && hist_mem[bin_cnt] != 0) begin
                        cdf_min       <= new_cdf;
                        cdf_min_found <= 1'b1;
                    end

                    // Build LUT entry: out = (CDF(in) - CDF_min) / (NPIX - CDF_min) * 255
                    // Use integer arithmetic: multiply first to avoid precision loss
                    // lut_val = ((new_cdf - cdf_min) * 255) / (NPIX - cdf_min)
                    // Guard against divide-by-zero (flat histogram → all zeros)
                    if (cdf_min_found && (NPIX - cdf_min) > 0)
                        lut_val = ((new_cdf - cdf_min) * 255) / (NPIX - cdf_min);
                    else
                        lut_val = '0;

                    lut_mem[bin_cnt] <= (lut_val > 255) ? 8'd255 : lut_val[W_I-1:0];

                    if (bin_cnt == 8'hFF) begin
                        pix_cnt <= '0;
                        state   <= MAP;
                    end else begin
                        bin_cnt <= bin_cnt + 1;
                    end
                end

                // ----------------------------------------------------------
                // MAP pass: apply LUT to each pixel
                MAP: begin
                    pix_out[pix_cnt] <= lut_mem[pix_buf[pix_cnt]];
                    if (pix_cnt == NPIX-1) begin
                        state <= EMIT;
                    end else begin
                        pix_cnt <= pix_cnt + 1;
                    end
                end

                // ----------------------------------------------------------
                // EMIT: hold output until downstream accepts
                EMIT: begin
                    m_valid <= 1'b1;
                    if (m_valid && m_ready) begin
                        m_valid <= 1'b0;
                        s_ready <= 1'b1;
                        state   <= IDLE;
                    end
                end

            endcase
        end
    end

    // Pack output pixel array to flat bus
    generate
        genvar gi;
        for (gi = 0; gi < NPIX; gi++) begin : pack_out
            assign m_data[gi*W_I +: W_I] = pix_out[gi];
        end
    endgenerate

endmodule
