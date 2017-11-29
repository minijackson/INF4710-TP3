constant sampler_t sampler =
		CLK_NORMALIZED_COORDS_FALSE |
		CLK_ADDRESS_CLAMP_TO_EDGE |
		CLK_FILTER_NEAREST;

kernel void dilate(
		read_only image2d_t input,
		private const int struct_el_radius,
		write_only image2d_t output) {

	int2 coord = (int2)(get_global_id(0), get_global_id(1));

	for(int i = -struct_el_radius; i <= struct_el_radius; ++i) {
		for(int j = -struct_el_radius; j <= struct_el_radius; ++j) {
			int2 offset = (int2)(i, j);

			if(read_imageui(input, sampler, coord + offset).x != 0) {
				write_imageui(output, coord, 255);
				return;
			}

		}
	}

	write_imageui(output, coord, 0);
}

