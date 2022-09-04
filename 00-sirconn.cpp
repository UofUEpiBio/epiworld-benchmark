#include "epiworld.hpp"
#include "bmark.hpp"

int main() {
	int n = 1000;



	double total = 0.0;
	REPLICATE(n)	
	{	
	TIME_START(model)
	epiworld::epimodels::ModelSIRCONN<> model(
			"VIRUS",
			10000,
			0.01,
			2,
			.9,
			.3
			);

	model.init(50, 1010);
	model.verbose_off();
	model.run();
	TIME_END(model)
	total += ELAPSED(model);
	}

	
	
	printf("\nElapsed time (%i reps): %.4f\n", n, total/n);


	return 0;

}
