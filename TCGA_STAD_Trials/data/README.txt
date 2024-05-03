sK-Medoids data integration, and preliminary analysis methods


<home directory>/pipeline.sh - Runs the entire integration process
Usage Examples:

MixOmics embedding GI subtype:
   bash pipeline.sh -t STAD -name STAD_GI_Subtype -st Subtype_Selected -mode MixOmics -ncomp 5

CustOmics embedding GI subtype:
   bash pipeline.sh -t STAD -name STAD_GI_Subtype -st Subtype_Selected -mode CustOmics -nlatent 20

MixOmics embedding Immune subtype:
   bash pipeline.sh -t STAD -name STAD_Immune_Subtype -st Immune.Subtype -mode MixOmics -ncomp 5

MixOmics embedding Immune subtype
   bash pipeline.sh -t STAD -name STAD_Immune_Subtype -st Immune.Subtype -mode CustOmics -nlatent 20

Note: When running immune subtypes: need to add Immune.Subtypes column (needs to match in .json) containing. Optionally perform mixomics completely unsupervised and labeling is ignored.

<home directory>/runTest.ipynb - Performs clustering including sK-Medoids and creates the graphs from integrated data


<home directory>/src

TCGA_Config.json - integration config variables for full analysis (desc. in code)TCGA_Config-DS.json - integration config variables for downsampled analysis (desc. in code)runMixOmics.r - Runs MixOmics dimensional reductionrunCustomicsMixed.py - Runs Customics dimensional reductionintegrate_data.py - Runs python integrationhelper_scripts.py - Contains helper scripts for integration, etc.download_TCGA.r - Downloads TCGA-STAD data or combination (unharmonized!)
process_TCGA.r - Empty script - TBD. May harmonize or further preprocess or harmonize data, or handle new types of datacustomics.py - Contains adapted source code for Customics - No survival, adapted for our purposes - minimal changes


Citations in comments!


Dependencies:

mixOmics (R)
TCGAbiolinks (R)
SummarizedExperiment (R)


sklearn (Python)
pickle (Python)
torch (Python)
pandas (Python)
numpy (Python)
scipy (Python)
shap (Python)
matplotlib (Python)
seaborn (Python)
collections (Python)