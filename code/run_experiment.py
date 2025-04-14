import data_pipeline.prepareData as pd
data = pd.Data(indicators=['cpi', 'gdp', 'debtgdp'])
data.getReady('CrisisData')

import doExperiment as de
exp = de.Experiment(data, ['Logit', 'RandomForest'], 'CrossVal')
exp.run()
