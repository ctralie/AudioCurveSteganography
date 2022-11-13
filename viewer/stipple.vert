attribute vec3 a_info;

// Uniforms set from Javascript that are constant
// over all fragments
uniform vec2 uCenter; // Where the origin (0, 0) is on the canvas
uniform float uScale; // Scale of fractal
uniform float uPointSize; // Size to draw points

varying float v_time;

void main() {
  gl_PointSize = uPointSize;
  vec2 a_position = vec2(a_info.x, a_info.y);
  gl_Position = vec4((a_position-uCenter)*uScale, 0, 1);
  v_time = a_info.z;
}