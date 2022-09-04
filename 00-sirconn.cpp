#include "epiworld.hpp"
#include "bmark.hpp"

int main() {

	TIME_START(con);
	epiworld::epimodels::ModelSIRCONN<> model(
			"VIRUS",
			10000,
			0.01,
			2,
			.9,
			.3
			);

	model.init(50, 1010);
	model.run();
	model.print();

	TIME_END(con);
	
	printf("\nElapsed time (%i reps): %.4f\n", 1, ELAPSED(con));


	return 0;

}
