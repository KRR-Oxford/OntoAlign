GS_file = 'GS_mappings_path_checked.txt'

LogMap_ML_annotated_sample_file = 'LogMap-ML_samples_a.txt'
LogMap_annotated_sample_file = 'LogMap_samples_a.txt'
AML_annotated_sample_file = 'AML_samples_a.txt'

LogMap_ML_output_mapping_file = 'LogMap-ML_output_mappings.txt'
LogMap_output_mapping_file = 'LogMap_mappings.txt'
AML_output_mapping_file = 'AML_mappings.txt'

GS = set()
with open(GS_file) as f:
    for line in f.readlines():
        GS.add(line.strip())

LML_mappings = set()
for line in open(LogMap_ML_output_mapping_file).readlines():
    LML_mappings.add(line.strip())
LML_samples = set()
with open(LogMap_ML_annotated_sample_file) as f:
    lines = [line.strip() for line in f.readlines()]
    for i in range(0, len(lines), 3):
        LML_samples.add(lines[i])
sv_n = 0
s_n = 0
for sample in LML_samples:
    tmp = sample.split('|')
    if '%s|%s' % (tmp[0], tmp[1]) not in GS:
        s_n += 1
        if len(tmp) >= 4 and tmp[3] == 'true':
            sv_n += 1
print('LogMap-ML: sampled mappings not in GS: %d, correct samples: %d, sampled precision: %f' % (s_n, sv_n, sv_n / s_n))

app_TP = len(LML_mappings.intersection(GS)) + len(LML_mappings - GS) * (sv_n / s_n)
app_p = app_TP / len(LML_mappings)
print('the approximate precision of LogMap-ML: %.3f' % app_p)


samples = set()
filenames = [LogMap_ML_annotated_sample_file, LogMap_annotated_sample_file, AML_annotated_sample_file]
for filename in filenames:
    with open(filename) as f:
        lines = [line.strip() for line in f.readlines()]
        for i in range(0, len(lines), 3):
            samples.add(lines[i])
sv_n = 0
s_n = 0
for sample in samples:
    tmp = sample.split('|')
    if '%s|%s' % (tmp[0], tmp[1]) not in GS:
        s_n += 1
        if len(tmp) >= 4 and tmp[3] == 'true':
            sv_n += 1

print('All three systems: sampled mappings not in GS: %d, correct samples: %d, sampled precision: %f' % (s_n, sv_n, sv_n / s_n))

from lib.Label import uri_prefix

mappings = set()
for line in open(LogMap_output_mapping_file).readlines():
    tmp = line.strip().split('|')
    c1 = uri_prefix(uri=tmp[0])
    c2 = uri_prefix(uri=tmp[1])
    mappings.add('%s|%s' % (c2, c2))
for line in open(LogMap_ML_output_mapping_file).readlines():
    mappings.add(line.strip())
for line in open(AML_output_mapping_file).readlines():
    mappings.add(line.strip())
G_M_n = len(mappings - GS)
print('mappings out of GS: %d' % G_M_n)

num = len(GS) + sv_n / s_n * G_M_n
print('approximate GS size: %d' % num)
print('approximate recall: %.3f' % (app_TP/num))
