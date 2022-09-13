#include "epiworld.hpp"
#include "bmark.hpp"

int main(int argc, char * argv[]) {


  int nsim = strtol(argv[1], nullptr, 0);
	int npop = strtol(argv[2], nullptr, 0);

	double total = 0.0;
	REPLICATE(nsim)	
	{	

  	TIME_START(model)
  	epiworld::epimodels::ModelSIRCONN<> model(
		  	"VIRUS",
		  	npop,
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

    if (i == (nsim - 1))
      model.print();

	}

	
	//printf("\nElapsed time (%i reps, %i size): %.4f ms\n", nsim, npop, total/nsim);


	return 0;

}
