import pandas as pd
import io
from google.colab import files
from scipy.stats import shapiro, levene, ttest_ind, mannwhitneyu

uploaded = files.upload()
df = pd.read_csv(io.BytesIO(list(uploaded.values())[0]))

metrics = df.columns.drop(['Group'])  # Group 열 이름 제외한 나머지 저장
results = []

for metric in metrics:
    intervention_group = df[df['Group'] == 'Intervention'][metric]
    control_group = df[df['Group'] == 'Control'][metric]

    if len(intervention_group) > 2 and len(control_group) > 2:
        shapiro_intervention = shapiro(intervention_group)
        shapiro_control = shapiro(control_group)

        levene_result = levene(intervention_group, control_group)

        # 정규분포 검사
        if shapiro_intervention.pvalue > 0.05:
            print(f"Intervention 그룹의 {metric} P-value는 {shapiro_intervention.pvalue:.4f}로, 정규분포한다.")
        else:
            print(f"Intervention 그룹의 {metric} P-value는 {shapiro_intervention.pvalue:.4f}로, 정규분포하지 않는다.")

        if shapiro_control.pvalue > 0.05:
            print(f"Control 그룹의 {metric} P-value는 {shapiro_control.pvalue:.4f}로, 정규분포한다.")
        else:
            print(f"Control 그룹의 {metric} P-value는 {shapiro_control.pvalue:.4f}로, 정규분포하지 않는다.")

        # 등분산 검사
        if levene_result.pvalue > 0.05:
            print(f"두 그룹의 {metric} 등분산 검정 결과는 {levene_result.pvalue:.4f}로, 등분산하다.")
        else:
            print(f"두 그룹의 {metric} 등분산 검정 결과는 {levene_result.pvalue:.4f}로, 등분산하지 않는다.")

        # 평균, 표준편차
        intervention_mean = intervention_group.mean()
        intervention_std = intervention_group.std()
        control_mean = control_group.mean()
        control_std = control_group.std()

        # 정규분포하다.
        if shapiro_intervention.pvalue > 0.05 and shapiro_control.pvalue > 0.05:
            # 등분산하다.
            if levene_result.pvalue > 0.05:
                test_name = 'T-test'
                test_result = ttest_ind(intervention_group, control_group, equal_var=True)
            # 등분산하지 않는다.
            else:
                test_name = 'Welch\'s t-test'
                test_result = ttest_ind(intervention_group, control_group, equal_var=False)
        # 정규분포하지 않는다.
        else:
            test_name = 'Mann-Whitney U test'
            test_result = mannwhitneyu(intervention_group, control_group, alternative='two-sided')

        statistic = test_result.statistic
        p_value = test_result.pvalue

        results.append({
            'Test Used': test_name,
            'Variables': metric,
            'Intervention(mean ± std)': f"{intervention_mean:.2f} ± {intervention_std:.2f}",
            'Control(mean ± std)': f"{control_mean:.2f} ± {control_std:.2f}",
            'Test Statistic': f"{statistic:.4f}",
            'P-value': f"{p_value:.4f}"
        })

results_df = pd.DataFrame(results)
results_df