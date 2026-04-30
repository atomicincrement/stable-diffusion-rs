// Compare noise schedules between our implementation and reference
// Run with: cargo run --release --example compare_noise

fn sep(n: usize) -> String {
    "=".repeat(n)
}

fn main() {
    let sep = sep(70);
    println!("{}", sep);
    println!("NOISE SCHEDULE COMPARISON: Our Implementation vs. Hugging Face");
    println!("{}", sep);
    println!();

    // Our Linear Schedule
    println!("OUR LINEAR SCHEDULE:");
    println!("  beta_start: 0.0001");
    println!("  beta_end: 0.02");
    println!("  Formula: β_t = 0.0001 + (0.02 - 0.0001) * t / 999");
    println!();
    
    let mut ours_linear_betas = vec![];
    for t in 0..1000 {
        let beta = 0.0001 + (0.02 - 0.0001) * (t as f32) / 999.0;
        ours_linear_betas.push(beta);
    }
    
    print_schedule_values("Ours Linear", &ours_linear_betas);
    println!();
    
    // Our Cosine Schedule  
    println!("OUR COSINE SCHEDULE:");
    println!("  Formula: ᾱ_t = (cos(π * t / 2000))^2");
    println!("  β_t = 1 - ᾱ_t / ᾱ_(t-1)");
    println!();
    
    let mut ours_cosine_betas = vec![];
    let s = 0.008;
    for i in 0..1000 {
        let t = (i as f32) / 1000.0;
        let alpha_bar = ((std::f32::consts::PI / 2.0 * t).cos()).powi(2);
        let alpha_bar_prev = if i == 0 {
            1.0
        } else {
            let t_prev = ((i - 1) as f32) / 1000.0;
            ((std::f32::consts::PI / 2.0 * t_prev).cos()).powi(2)
        };
        let beta = 1.0 - (alpha_bar / (alpha_bar_prev + s) * (1.0 - s)).min(0.999);
        ours_cosine_betas.push(beta.max(0.0001));
    }
    
    print_schedule_values("Ours Cosine", &ours_cosine_betas);
    println!();

    // Hugging Face DDPM (ScaledLinear)
    println!("HUGGING FACE DDPM (ScaledLinear = sqrt linear):");
    println!("  beta_start: 0.00085");
    println!("  beta_end: 0.012");
    println!("  Formula: β_t = (sqrt(0.00085) + (sqrt(0.012) - sqrt(0.00085)) * t / 999)^2");
    println!();
    
    let beta_start = 0.00085_f32.sqrt();
    let beta_end = 0.012_f32.sqrt();
    let mut hf_scaled_linear_betas = vec![];
    for t in 0..1000 {
        let interp = beta_start + (beta_end - beta_start) * (t as f32) / 999.0;
        let beta = interp * interp;
        hf_scaled_linear_betas.push(beta);
    }
    
    print_schedule_values("HF ScaledLinear", &hf_scaled_linear_betas);
    println!();

    // Hugging Face SquaredcosCapV2
    println!("HUGGING FACE SquaredcosCapV2 (Cosine variant):");
    println!("  Formula: ᾱ_t = (cos((t + 0.008) / 1.008 * π/2))^2");
    println!();
    
    let mut hf_cosine_betas = vec![];
    for i in 0..1000 {
        let alpha_bar = ((((i as f64 + 0.008) / 1.008) * std::f64::consts::FRAC_PI_2).cos()).powi(2);
        let alpha_bar_prev = if i == 0 {
            1.0
        } else {
            (((((i - 1) as f64 + 0.008) / 1.008) * std::f64::consts::FRAC_PI_2).cos()).powi(2)
        };
        let beta = (1.0 - alpha_bar / alpha_bar_prev).min(0.999);
        hf_cosine_betas.push(beta as f32);
    }
    
    print_schedule_values("HF SquaredCosV2", &hf_cosine_betas);
    println!();

    // Compare at specific timesteps
    println!("{}", sep);
    println!("TIMESTEP COMPARISON (key points):");
    println!("{}", sep);
    println!();
    
    println!("{:6} | {:12} | {:12} | {:12} | {:12}", 
             "Step", "Ours Lin", "HF ScaledLin", "Ours Cos", "HF SquaredCos");
    println!("{:6}-+{:12}-+{:12}-+{:12}-+{:12}", "-", "-", "-", "-", "-");
    
    for &step in &[0, 1, 10, 100, 250, 500, 750, 999] {
        println!("{:6} | {:12.6} | {:12.6} | {:12.6} | {:12.6}",
                 step,
                 ours_linear_betas[step],
                 hf_scaled_linear_betas[step],
                 ours_cosine_betas[step],
                 hf_cosine_betas[step]);
    }
    println!();

    // Summary
    println!("{}", sep);
    println!("KEY DIFFERENCES:");
    println!("{}", sep);
    println!();
    println!("1. BETA RANGE:");
    println!("   Our Linear:     [0.0001, 0.02]");
    println!("   HF ScaledLin:   [0.00085, 0.012]");
    println!("   → HF uses lower noise range (more conservative)");
    println!();
    println!("2. LINEAR VS SCALED LINEAR:");
    println!("   Our Linear:     Direct interpolation");
    println!("   HF ScaledLin:   sqrt interpolation (slower at start)");
    println!();
    println!("3. COSINE SCHEDULES:");
    println!("   Our Cosine:     Standard formula");
    println!("   HF SquaredCosV2: Slightly different offset (0.008/1.008)");
    println!();
    println!("RECOMMENDATION:");
    println!("  For best results matching SD 1.5, use:");
    println!("  - beta_start: 0.00085");
    println!("  - beta_end: 0.012");
    println!("  - beta_schedule: ScaledLinear (sqrt interpolation)");
    println!();
}

fn print_schedule_values(_name: &str, betas: &[f32]) {
    let min = betas.iter().cloned().fold(f32::INFINITY, f32::min);
    let max = betas.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let mean = betas.iter().sum::<f32>() / betas.len() as f32;
    
    println!("  Min β: {:.8}", min);
    println!("  Max β: {:.8}", max);
    println!("  Mean β: {:.8}", mean);
    println!("  β[0]:   {:.8}", betas[0]);
    println!("  β[500]: {:.8}", betas[500]);
    println!("  β[999]: {:.8}", betas[999]);
}
