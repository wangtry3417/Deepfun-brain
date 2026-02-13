#include <stdio.h>
#include <ml.h>
#include <fungpt.w>

set model of funGPT() into mainStage;

func main() {
   model.set(**ml.MODELSETTINGS);
   model.fit(ds=ml.datasets.DEEPTEXTCHAT);
   model.train(ep=1000);
}