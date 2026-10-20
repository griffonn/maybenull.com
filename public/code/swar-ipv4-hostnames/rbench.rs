use std::net::Ipv4Addr; use std::time::Instant;
fn main(){ let f=std::env::args().nth(1).unwrap(); let d=std::fs::read_to_string(&f).unwrap(); let v:Vec<&str>=d.lines().collect();
 let mut best=f64::MAX; let mut sink=0u32;
 for _ in 0..7 { let t=Instant::now(); for s in &v { sink=sink.wrapping_add(match std::hint::black_box(*s).parse::<Ipv4Addr>(){Ok(a)=>u32::from(a),Err(_)=>1}); }
   best=best.min(t.elapsed().as_nanos() as f64/v.len() as f64); }
 println!("rust_std={:.2} ({})",best,sink); }
