class Layer{
	public:
		int M, N, O; // O: output, N: #feature, M: #params_per_feature
		float *pre_output, *output;
		float *weight, *bias;
		Layer(int M, int N, int O);
		~Layer();
};