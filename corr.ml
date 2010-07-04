let correlate corr_size base mask sample_size =
  let correlation = Array.create (corr_size*corr_size) 0.0 in
  for oy = 0 to corr_size-1 do
    for ox = 0 to corr_size-1 do
      let sum = ref 0.0 in
      for y = 0 to sample_size-oy-1 do
        let base_idx = ((y+oy)*sample_size+ox) * 4 in
        let mask_idx = (y * sample_size * 4) in
        for x = 0 to sample_size-ox-1 do
          let b = base_idx+4*x and m = mask_idx+4*x in
          sum := !sum +. (
            base.(b) *. mask.(m) +.
            base.(b+1) *. mask.(m+1) +.
            base.(b+2) *. mask.(m+2) +.
            base.(b+3) *. mask.(m+3)
          )
        done
      done;
      correlation.(oy*corr_size+ox) <- !sum
    done
  done;
  correlation

let mksz sz =
  Array.init (sz*sz*4) (fun i -> float i *. 0.000001)

let () =
  let base = mksz 128 in
  let mask = mksz 128 in
  ignore (correlate 64 base mask 128)
