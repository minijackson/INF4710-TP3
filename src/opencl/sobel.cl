constant sampler_t sampler =
		CLK_NORMALIZED_COORDS_FALSE |
		CLK_ADDRESS_CLAMP_TO_EDGE |
		CLK_FILTER_NEAREST;

int constant kernel_x[9] = {1, 0, -1, 2, 0, -2, 1, 0, -1};
int constant kernel_y[9] = {1, 2, 1, 0, 0, 0, -1, -2, -1};

kernel void sobel(
		read_only image2d_t input,
		write_only image2d_t output) {

	int2 coord = (int2)(get_global_id(0), get_global_id(1));

#if 0

	int kernel_index = 0;
	int3 grad_x = 0, grad_y = 0;

	for(int i = -1; i <= 1; ++i) {
		for(int j = -1; j <= 1; ++j) {
			int2 offset = (int2)(i, j);

			int3 pixel = convert_int3(read_imageui(input, sampler, coord + offset).xyz);
			grad_x += pixel * kernel_x[kernel_index];
			grad_y += pixel * kernel_y[kernel_index];

			++kernel_index;
		}
	}

#else

	int3 NW = convert_int3(read_imageui(input, sampler, coord + (int2)(-1, -1)).xyz),
		 N  = convert_int3(read_imageui(input, sampler, coord + (int2)(-1,  0)).xyz),
		 NE = convert_int3(read_imageui(input, sampler, coord + (int2)(-1,  1)).xyz),

		 E = convert_int3(read_imageui(input, sampler, coord + (int2)(0, -1)).xyz),
		 W = convert_int3(read_imageui(input, sampler, coord + (int2)(0,  1)).xyz),

		 SE = convert_int3(read_imageui(input, sampler, coord + (int2)(1, -1)).xyz),
		 S  = convert_int3(read_imageui(input, sampler, coord + (int2)(1,  0)).xyz),
		 SW = convert_int3(read_imageui(input, sampler, coord + (int2)(1,  1)).xyz);

	int3 grad_x = NW + SW + (2 * W) - NE - SE - (2 * E);
	int3 grad_y = NW + NE + (2 * N) - SW - SE - (2 * S);

#endif

	/*double3 dgrad_x = convert_double3(grad_x),*/
			/*dgrad_y = convert_double3(grad_y);*/

	double3 new_values = clamp(
			/*sqrt((dgrad_x * dgrad_x) + (dgrad_y * dgrad_y)),*/
			convert_double3((abs(grad_x) + abs(grad_y)) >> 1) * 1.414216,
			0.0,
			255.0
			);

	uint4 new_pixel = (uint4)(convert_uint3(new_values), 255);

	write_imageui(output, coord, new_pixel);
}
