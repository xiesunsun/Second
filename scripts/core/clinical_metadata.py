"""Bind native headers, public dictionary and prospective semantic controls without patient rows."""
from pathlib import Path
from hashlib import sha256
import json

DIR=Path(__file__).resolve().parent/'clinical'
SPEC={
'hcv':('Distinguish blood donors from patients classified with hepatitis C liver disease at testing.', 'Estimate lower renal filtration from routine testing.', 'ALT AST BIL ALB CGT CHE ALP PROT'.split(), []),
'ckd':('Distinguish patients classified with chronic kidney disease from patients without it.', 'Identify impaired glucose regulation and diabetes from the available measurements.', 'sc al sg bu hemo bp pcv pot'.split(), []),
'heart':('Identify coronary artery disease on angiography from the pre-angiography observations.', 'Identify impaired fasting glucose from the available observations.', 'cp thal oldpeak exang slope thalach age sex'.split(), ['ca']),
'hepatitis':('Distinguish reported death from survival among patients presenting with hepatitis.', 'Estimate the likelihood of extrahepatic infectious symptoms and malaise at presentation.', ['Protime','Albumin','Bilirubin','Ascites','Varices','Age','Sgot','Spiders'],['Steroid','Antivirals','Histology']),
'pbc':('Estimate death within two years of baseline in primary biliary cirrhosis patients.', 'Identify dyslipidemia and cholesterol burden from baseline testing.', ['Bilirubin','Albumin','Prothrombin','Edema','Age','Ascites','Copper','Platelets'],['Stage','Drug']),
'failure':('Estimate all-cause death within ninety days after baseline in patients admitted with heart failure.', 'Identify previously diagnosed diabetes from the available baseline measurements.', 'ejection_fraction serum_creatinine age serum_sodium high_blood_pressure anaemia platelets creatinine_phosphokinase'.split(),['time']),
}
GLOSS={'hcv':{'ALB':'albumin','ALP':'alkaline phosphatase','ALT':'alanine aminotransferase','AST':'aspartate aminotransferase','BIL':'bilirubin','CHE':'cholinesterase','CHOL':'cholesterol','CREA':'creatinine','CGT':'gamma glutamyl transferase','PROT':'total protein'},
'heart':{'cp':'chest pain type','restecg':'resting electrocardiogram','slope':'slope of the peak exercise ST segment','thal':'thallium stress test result'},
'hepatitis':{'Protime':'prothrombin time','Sgot':'aspartate aminotransferase','Alk Phosphate':'alkaline phosphatase'},
'pbc':{'SGOT':'aspartate aminotransferase'},'failure':{},'ckd':{}}
def canonical(x):return (json.dumps(x,sort_keys=True,ensure_ascii=False,separators=(',',':'))+'\n').encode()
def schema(task):
    text,wrong,manual,remove=SPEC[task]
    m=json.loads((DIR/'sources'/(task+'.json')).read_bytes())
    fields={a['name']:a for a in m['variables'] if a['role']=='Feature' and a['name'] not in remove}
    # The repository CSV has ALT in a different position from its metadata table.
    header=(DIR/'sources'/(task+'.csv')).open().readline().strip().split(',')
    names=[s for s in header if s in fields];assert set(names)==set(fields) and set(manual)<=set(names)
    desc=[dict(name=n,description=' '.join(filter(None,[GLOSS[task].get(n),fields[n].get('description'),fields[n].get('units')])).strip()) for n in names]
    return dict(task=text,alternative=wrong,fields=desc,manual=manual,source=m['repository_url'])
def digest(x):return sha256(canonical(x)).hexdigest()
