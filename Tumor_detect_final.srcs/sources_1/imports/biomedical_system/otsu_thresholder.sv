`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: Department of Computer Engineering, University of Peradeniya
// Engineer: Mahela Ekanayake, Chaminda Weerasinghe
//
// Module Name: otsu_thresholder.sv
// Description: Otsu automatic thresholding for binary segmentation.
//
//   Pass 1 (HIST):   Accumulate 256-bin histogram.
//   Pass 2 (OTSU):   Sweep t = 0..255, compute between-class variance
//                    σ²_B(t) = w0*w1*(μ0-μ1)²  using integer arithmetic.
//                    Track t* = argmax σ²_B(t).
//   Pass 3 (THRESH): Apply binary threshold: pix >= t* → 255, else → 0.
//
//   Special case: single-intensity tile → t* = 0, all pixels → 255.
//
// Dependencies: none
//////////////////////////////////////////////////////////////////////////////////

module otsu_thresholder #(
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

    localparam NPIX  = R_I * C_I;
    localparam NBINS = 256;

    // -----------------------------------------------------------------------
    // Block RAM: histogram
    // -----------------------------------------------------------------------
    logic [31:0] hist_mem [0:NBINS-1];

    // -----------------------------------------------------------------------
    // Pixel buffer
    // -----------------------------------------------------------------------
    logic [W_I-1:0] pix_buf [0:NPIX-1];
    logic [W_I-1:0] pix_out [0:NPIX-1];

    // -----------------------------------------------------------------------
    // FSM
    // -----------------------------------------------------------------------
    typedef enum logic [2:0] {
        IDLE, HIST, OTSU, THRESH, EMIT
    } state_t;

    state_t state;

    logic [$clog2(NPIX)-1:0] pix_cnt;
    logic [7:0]              bin_cnt;
    logic [7:0]              t_star;

    // Otsu accumulators (64-bit to avoid overflow for large tiles)
    logic [63:0] sum_total;   // sum of all intensity*count
    logic [31:0] w0;          // weight of class 0 (pixel count below t)
    logic [63:0] sum0;        // sum of intensities in class 0
    logic [63:0] best_var;    // best between-class variance seen so far

    // -----------------------------------------------------------------------
    // FSM + datapath
    // -----------------------------------------------------------------------
    always_ff @(posedge clk or negedge rstn) begin
        if (!rstn) begin
            state      <= IDLE;
            s_ready    <= 1'b1;
            m_valid    <= 1'b0;
            pix_cnt    <= '0;
            bin_cnt    <= '0;
            t_star     <= '0;
            sum_total  <= '0;
            w0         <= '0;
            sum0       <= '0;
            best_var   <= '0;
            for (int i = 0; i < NBINS; i++) hist_mem[i] <= '0;
        end else begin
            case (state)

                // ----------------------------------------------------------
                IDLE: begin
                    m_valid <= 1'b0;
                    if (s_valid && s_ready) begin
                        for (int i = 0; i < NPIX; i++)
                            pix_buf[i] <= s_data[i*W_I +: W_I];
                        for (int i = 0; i < NBINS; i++) hist_mem[i] <= '0;
                        s_ready   <= 1'b0;
                        pix_cnt   <= '0;
                        state     <= HIST;
                    end
                end

                // ----------------------------------------------------------
                HIST: begin
                    hist_mem[pix_buf[pix_cnt]] <= hist_mem[pix_buf[pix_cnt]] + 1;
                    if (pix_cnt == NPIX-1) begin
                        // Pre-compute total weighted sum for Otsu
                        sum_total <= '0;
                        pix_cnt   <= '0;
                        bin_cnt   <= '0;
                        w0        <= '0;
                        sum0      <= '0;
                        best_var  <= '0;
                        t_star    <= '0;
                        state     <= OTSU;
                    end else begin
                        pix_cnt <= pix_cnt + 1;
                    end
                end

                // ----------------------------------------------------------
                // First half of OTSU state: accumulate sum_total (256 cycles)
                // We reuse bin_cnt; after 256 cycles switch to sweep.
                // To keep it simple we do two sub-phases via a flag.
                OTSU: begin
                    automatic logic [63:0] var_num;
                    automatic logic [31:0] w1;
                    automatic logic [63:0] mu0_num, mu1_num;
                    automatic logic [63:0] diff;

                    // Accumulate sum_total in first pass (bin_cnt counts 0..255)
                    // We detect first pass by checking sum_total == 0 at bin_cnt==0
                    // Simpler: use a separate sub-state flag
                    // Implementation: two-pass using sum_total accumulation first
                    // then Otsu sweep. We use pix_cnt as a phase flag (0=sum, 1=sweep).

                    if (pix_cnt == 0) begin
                        // Phase 0: accumulate sum_total
                        sum_total <= sum_total + 64'(bin_cnt) * 64'(hist_mem[bin_cnt]);
                        if (bin_cnt == 8'hFF) begin
                            bin_cnt <= '0;
                            pix_cnt <= 1;   // switch to sweep phase
                        end else begin
                            bin_cnt <= bin_cnt + 1;
                        end
                    end else begin
                        // Phase 1: Otsu sweep
                        w0   <= w0   + hist_mem[bin_cnt];
                        sum0 <= sum0 + 64'(bin_cnt) * 64'(hist_mem[bin_cnt]);

                        w1 = NPIX - (w0 + hist_mem[bin_cnt]);

                        if (w1 > 0 && (w0 + hist_mem[bin_cnt]) > 0) begin
                            mu0_num = sum0 + 64'(bin_cnt) * 64'(hist_mem[bin_cnt]);
                            mu1_num = sum_total - mu0_num;
                            // var = w0*w1*(mu0 - mu1)^2
                            // Use scaled integer: multiply out, compare relative
                            // diff = |mu0*w1 - mu1*w0| (avoids division)
                            // var_proxy = diff^2 / (w0*w1) — but we compare
                            // w0*w1*(mu0_num/w0 - mu1_num/w1)^2
                            // = (mu0_num*w1 - mu1_num*w0)^2 / (w0*w1)
                            // For comparison purposes use numerator only:
                            // proxy = (mu0_num*w1 - mu1_num*w0)^2
                            if (mu0_num * w1 >= mu1_num * (w0 + hist_mem[bin_cnt]))
                                diff = mu0_num * w1 - mu1_num * (w0 + hist_mem[bin_cnt]);
                            else
                                diff = mu1_num * (w0 + hist_mem[bin_cnt]) - mu0_num * w1;
                            var_num = diff * diff;
                            if (var_num > best_var) begin
                                best_var <= var_num;
                                t_star   <= bin_cnt;
                            end
                        end

                        if (bin_cnt == 8'hFF) begin
                            pix_cnt <= '0;
                            state   <= THRESH;
                        end else begin
                            bin_cnt <= bin_cnt + 1;
                        end
                    end
                end

                // ----------------------------------------------------------
                THRESH: begin
                    pix_out[pix_cnt] <= (pix_buf[pix_cnt] >= t_star) ? 8'd255 : 8'd0;
                    if (pix_cnt == NPIX-1) begin
                        state <= EMIT;
                    end else begin
                        pix_cnt <= pix_cnt + 1;
                    end
                end

                // ----------------------------------------------------------
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

    // Pack output
    generate
        genvar gi;
        for (gi = 0; gi < NPIX; gi++) begin : pack_out
            assign m_data[gi*W_I +: W_I] = pix_out[gi];
        end
    endgenerate

endmodule
