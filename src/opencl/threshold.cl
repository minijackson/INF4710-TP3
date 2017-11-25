constant sampler_t sampler =
		CLK_NORMALIZED_COORDS_FALSE |
		CLK_ADDRESS_CLAMP_TO_EDGE |
		CLK_FILTER_NEAREST;

uint get_value(uint3 rgb) {
	return max(max(rgb.x, rgb.y), rgb.z);
}

uint get_intensity(uint3 rgb) {
	return (rgb.x + rgb.y + rgb.z) / 3;
}

kernel void threshold(
		read_only image2d_t input,
		private const uchar limit,
		write_only image2d_t output){
	int2 coord = (int2)(get_global_id(0), get_global_id(1));

	uint4 pixel = read_imageui(input, sampler, coord);
	/*uint value = get_value(pixel.xyz);*/
	uint intensity = get_intensity(pixel.xyz);

	write_imageui(output, coord, (intensity > limit) ? 255 : 0);
}
