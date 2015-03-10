float multiply(float Cb, float Cs) { return Cb * Cs; }

float screen(float Cb, float Cs) { return Cb + Cs - (Cb * Cs); }

float hard_light(float Cb, float Cs) {
  if (Cs <= 0.5)
    return multiply(Cb, 2 * Cs);
  else
    return screen(Cb, 2 * Cs - 1);
}

float color_dodge(float Cb, float Cs) {
  if (Cb == 0)
    return 0;
  else if (Cs == 1)
    return 1;
  else
    return min(1.0, Cb / (1.0 - Cs));
}

float color_burn(float Cb, float Cs) {
  if (Cb == 1)
    return 1;
  else if (Cs == 0)
    return 0;
  else
    return 1 - min(1.0, (1.0 - Cb) / Cs);
}

float soft_light(float Cb, float Cs) {
  float d = 0;

  if (Cb <= 0.25)
    d = ((16 * Cb - 12) * Cb + 4) * Cb;
  else
    d = sqrt(Cb);

  if (Cs <= 0.5)
    return Cb - (1 - 2 * Cs) * Cb * (1 - Cb);
  else
    return Cb + (2 * Cs - 1) * (d - Cb);
}

kernel void blend_multiply(global float *Cb, global float *Cs,
                           global float *B) {
  size_t i = get_global_id(0);

  B[i] = multiply(Cb[i], Cs[i]);
}

kernel void blend_screen(global float *Cb, global float *Cs, global float *B) {
  size_t i = get_global_id(0);

  B[i] = screen(Cb[i], Cs[i]);
}

kernel void blend_normal(global float *Cb, global float *Cs, global float *B) {
  size_t i = get_global_id(0);

  B[i] = Cs[i];
}

kernel void blend_overlay(global float *Cb, global float *Cs, global float *B) {
  size_t i = get_global_id(0);

  B[i] = hard_light(Cs[i], Cb[i]);
}

kernel void blend_darken(global float *Cb, global float *Cs, global float *B) {
  size_t i = get_global_id(0);

  B[i] = min(Cb[i], Cs[i]);
}

kernel void blend_lighten(global float *Cb, global float *Cs, global float *B) {
  size_t i = get_global_id(0);

  B[i] = max(Cb[i], Cs[i]);
}

kernel void blend_color_dodge(global float *Cb, global float *Cs,
                              global float *B) {
  size_t i = get_global_id(0);

  B[i] = color_dodge(Cb[i], Cs[i]);
}

kernel void blend_hard_light(global float *Cb, global float *Cs,
                             global float *B) {
  size_t i = get_global_id(0);

  B[i] = hard_light(Cb[i], Cs[i]);
}

kernel void blend_soft_light(global float *Cb, global float *Cs,
                             global float *B) {
  size_t i = get_global_id(0);

  B[i] = soft_light(Cb[i], Cs[i]);
}

kernel void blend_difference(global float *Cb, global float *Cs,
                             global float *B) {
  size_t i = get_global_id(0);

  B[i] = fabs(Cb[i] - Cs[i]);
}

kernel void blend_exclusion(global float *Cb, global float *Cs,
                            global float *B) {
  size_t i = get_global_id(0);

  B[i] = Cb[i] + Cs[i] - 2 * Cb[i] * Cs[i];
}
