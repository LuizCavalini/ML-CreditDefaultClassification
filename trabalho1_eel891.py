"""
================================================================================
 EEL891 - Introdução ao Aprendizado de Máquina (2025-2)
 Trabalho 1 - Classificação de Inadimplência de Crédito (Kaggle)
 Aluno: Luiz Felipe Píccoli Cavalini
================================================================================
 SUMÁRIO: 1.EDA  2.Pré-proc  3.Features  4.Encoding  5.Baselines
          6.Hiperparâmetros  7.CV Final  8.Submissão Blend  9.Feature Importance
 INSTALAR: pip install pandas numpy matplotlib seaborn scikit-learn lightgbm xgboost catboost
================================================================================
"""
import pandas as pd, numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt, seaborn as sns, warnings
warnings.filterwarnings('ignore')
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import accuracy_score
import lightgbm as lgb, xgboost as xgb
from catboost import CatBoostClassifier
SEED = 42; np.random.seed(SEED)

# ====================== 1. EDA ======================
print("="*70+"\n 1. CARREGAMENTO E EDA\n"+"="*70)
train = pd.read_csv('conjunto_de_treinamento.csv')
test = pd.read_csv('conjunto_de_teste.csv')
print(f"  Treino: {train.shape} | Teste: {test.shape}")
print(f"  Target: {dict(train['inadimplente'].value_counts())} -> 50/50 balanceado")

nulls = train.isnull().sum()
print("  Nulos:", {c: n for c, n in nulls[nulls > 0].items()})

num_cols = train.select_dtypes(include=[np.number]).columns.drop(
    ['id_solicitante', 'inadimplente', 'grau_instrucao'])
corrs = train[num_cols].corrwith(train['inadimplente']).abs().sort_values(ascending=False)
print("  Top correlações:", {c: round(v, 3) for c, v in corrs.head(5).items()})

# Visualizações
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
fig.suptitle('EDA - Variáveis vs Inadimplência', fontsize=14)
for label, color in [(0, 'steelblue'), (1, 'salmon')]:
    sub = train[train['inadimplente'] == label]
    axes[0,0].hist(sub['idade'], bins=40, alpha=0.5,
                    label=f'{"Bom" if label==0 else "Mau"}', color=color)
axes[0,0].legend(); axes[0,0].set_title('Idade')
train.groupby(['dia_vencimento','inadimplente']).size().unstack(fill_value=0).plot(
    kind='bar', ax=axes[0,1], color=['steelblue','salmon'])
axes[0,1].set_title('Dia Venc.'); axes[0,1].tick_params(axis='x', rotation=0)
for label, color in [(0, 'steelblue'), (1, 'salmon')]:
    sub = train[train['inadimplente'] == label]
    axes[0,2].hist(np.log1p(sub['renda_mensal_regular']), bins=40, alpha=0.5,
                    label=f'{"Bom" if label==0 else "Mau"}', color=color)
axes[0,2].legend(); axes[0,2].set_title('log(Renda)')
train.groupby(['estado_civil','inadimplente']).size().unstack(fill_value=0).plot(
    kind='bar', ax=axes[1,0], color=['steelblue','salmon'])
axes[1,0].set_title('Estado Civil'); axes[1,0].tick_params(axis='x', rotation=0)
train.groupby(['forma_envio_solicitacao','inadimplente']).size().unstack(fill_value=0).plot(
    kind='bar', ax=axes[1,1], color=['steelblue','salmon'])
axes[1,1].set_title('Forma Envio'); axes[1,1].tick_params(axis='x', rotation=0)
train.groupby(['vinculo_formal_com_empresa','inadimplente']).size().unstack(fill_value=0).plot(
    kind='bar', ax=axes[1,2], color=['steelblue','salmon'])
axes[1,2].set_title('Vínculo'); axes[1,2].tick_params(axis='x', rotation=0)
plt.tight_layout(); plt.savefig('eda_visualizacoes.png', dpi=150); plt.close()
print("  Gráficos: eda_visualizacoes.png")

# ====================== 2. PRÉ-PROCESSAMENTO ======================
print("\n"+"="*70+"\n 2. PRÉ-PROCESSAMENTO\n"+"="*70)
target = train['inadimplente'].copy()
test_ids = test['id_solicitante'].copy()
train['is_train'] = 1; test['is_train'] = 0; test['inadimplente'] = -1
df = pd.concat([train, test], ignore_index=True)
df = df.drop(columns=['id_solicitante', 'grau_instrucao', 'possui_telefone_celular',
                       'qtde_contas_bancarias_especiais', 'local_onde_trabalha'])

df['sexo'] = df['sexo'].astype(str).str.strip().replace({'': 'N', 'nan': 'N'})
df['estado_onde_nasceu'] = df['estado_onde_nasceu'].astype(str).str.strip().replace(
    {'': 'DESC', 'XX': 'DESC', 'nan': 'DESC'})
df['estado_onde_reside'] = df['estado_onde_reside'].astype(str).str.strip()
df['estado_onde_trabalha'] = df['estado_onde_trabalha'].astype(str).str.strip().replace(
    {'': 'NAO_INF', 'nan': 'NAO_INF'})
df['forma_envio_solicitacao'] = df['forma_envio_solicitacao'].astype(str).str.strip()
for c in ['codigo_area_telefone_residencial', 'codigo_area_telefone_trabalho']:
    df[c] = df[c].astype(str).str.strip().replace({'': 'SEM', 'nan': 'SEM'})
for c in ['possui_telefone_residencial', 'vinculo_formal_com_empresa', 'possui_telefone_trabalho']:
    df[c] = (df[c].astype(str).str.strip() == 'Y').astype(int)
df['meses_na_residencia'] = df['meses_na_residencia'].fillna(df['meses_na_residencia'].median())
df['tipo_residencia'] = df['tipo_residencia'].fillna(df['tipo_residencia'].mode()[0])
df['profissao'] = df['profissao'].fillna(-1)
df['ocupacao'] = df['ocupacao'].fillna(-1)
df['tem_companheiro'] = (~df['profissao_companheiro'].isnull()).astype(int)
df['profissao_companheiro'] = df['profissao_companheiro'].fillna(-1)
df['grau_instrucao_companheiro'] = df['grau_instrucao_companheiro'].fillna(-1)
print("  5 colunas removidas, strings tratadas, nulos imputados.")

# ====================== 3. FEATURE ENGINEERING ======================
print("\n"+"="*70+"\n 3. FEATURE ENGINEERING\n"+"="*70)
df['renda_total'] = df['renda_mensal_regular'] + df['renda_extra']
df['tem_renda_extra'] = (df['renda_extra'] > 0).astype(int)
df['razao_patrimonio_renda'] = df['valor_patrimonio_pessoal'] / (df['renda_mensal_regular'] + 1)
cc = ['possui_cartao_visa', 'possui_cartao_mastercard', 'possui_cartao_diners',
      'possui_cartao_amex', 'possui_outros_cartoes']
df['qtde_cartoes'] = df[cc].sum(axis=1)
df['tem_algum_cartao'] = (df['qtde_cartoes'] > 0).astype(int)
df['reside_mesmo_estado'] = (df['estado_onde_nasceu'] == df['estado_onde_reside']).astype(int)
df['trabalha_mesmo_estado'] = (df['estado_onde_trabalha'] == df['estado_onde_reside']).astype(int)
df['qtde_telefones'] = df['possui_telefone_residencial'] + df['possui_telefone_trabalho']
df['faixa_etaria'] = pd.cut(df['idade'], bins=[0,25,35,45,55,65,200],
                             labels=[0,1,2,3,4,5]).astype(int)
df['log_renda'] = np.log1p(df['renda_mensal_regular'])
df['log_patrimonio'] = np.log1p(df['valor_patrimonio_pessoal'])
df['log_renda_total'] = np.log1p(df['renda_total'])
df['idade_x_renda'] = df['idade'] * df['log_renda']
df['vinculo_x_renda'] = df['vinculo_formal_com_empresa'] * df['log_renda']
print(f"  14 features criadas. Total: {df.shape[1]} colunas.")

# ====================== 4. ENCODING ======================
print("\n"+"="*70+"\n 4. ENCODING\n"+"="*70)
for c in ['sexo', 'forma_envio_solicitacao']:
    df[c] = LabelEncoder().fit_transform(df[c])
train_mask = df['is_train'] == 1; gm = target.mean()
for c in ['estado_onde_nasceu', 'estado_onde_reside', 'estado_onde_trabalha',
           'codigo_area_telefone_residencial', 'codigo_area_telefone_trabalho']:
    t = df.loc[train_mask, [c]].copy(); t['tgt'] = target.values
    means = t.groupby(c)['tgt'].mean()
    df[f'{c}_te'] = df[c].map(means).fillna(gm)
    df = df.drop(columns=[c])
for c in ['profissao', 'ocupacao', 'profissao_companheiro', 'grau_instrucao_companheiro',
           'tipo_residencia', 'estado_civil', 'produto_solicitado', 'dia_vencimento',
           'nacionalidade', 'tipo_endereco']:
    df[c] = df[c].astype(float)

feat_cols = [c for c in df.columns if c not in ['is_train', 'inadimplente']]
X = df.loc[df['is_train'] == 1, feat_cols].values.astype(np.float64)
y = target.values
X_test = df.loc[df['is_train'] == 0, feat_cols].values.astype(np.float64)
feat_names = feat_cols
print(f"  X: {X.shape}, X_test: {X_test.shape}, NaN: {np.isnan(X).sum()}")

# ====================== 5. BASELINES ======================
print("\n"+"="*70+"\n 5. BASELINES — 5-Fold CV\n"+"="*70)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
for name, m in [
    ('LogReg', LogisticRegression(max_iter=1000, random_state=SEED)),
    ('RF', RandomForestClassifier(n_estimators=200, max_depth=10,
                                   random_state=SEED, n_jobs=-1)),
    ('LGBM', lgb.LGBMClassifier(n_estimators=400, max_depth=6, learning_rate=0.05,
                                 verbose=-1, random_state=SEED)),
    ('XGB', xgb.XGBClassifier(n_estimators=400, max_depth=6, learning_rate=0.05,
                               random_state=SEED, eval_metric='logloss', verbosity=0)),
    ('CatBoost', CatBoostClassifier(iterations=400, depth=6, learning_rate=0.05,
                                     random_seed=SEED, verbose=0))]:
    s = cross_val_score(m, X, y, cv=cv, scoring='accuracy')
    print(f"  {name:12s} -> {s.mean():.4f} +/- {s.std():.4f}")

# ====================== 6. HIPERPARÂMETROS OTIMIZADOS (Optuna) ======================
print("\n"+"="*70+"\n 6. HIPERPARÂMETROS OTIMIZADOS (Optuna, split 80/20)\n"+"="*70)

lgbm_p = {
    'n_estimators': 1026, 'max_depth': 5, 'num_leaves': 65,
    'learning_rate': 0.027231, 'min_child_samples': 69,
    'subsample': 0.87495, 'colsample_bytree': 0.896608,
    'reg_alpha': 1.92e-07, 'reg_lambda': 0.002559,
}
xgb_p = {
    'n_estimators': 310, 'max_depth': 4,
    'learning_rate': 0.049019, 'min_child_weight': 2,
    'subsample': 0.646798, 'colsample_bytree': 0.59572,
    'gamma': 9.76e-07, 'reg_alpha': 2.13e-07, 'reg_lambda': 1.12e-07,
}
cat_p = {
    'iterations': 400, 'depth': 7,
    'learning_rate': 0.088106, 'l2_leaf_reg': 3.19055,
    'bagging_temperature': 0.921874,
}
print("  LGBM:", lgbm_p)
print("  XGB: ", xgb_p)
print("  CAT: ", cat_p)

# ====================== 7. CV FINAL + ENSEMBLE ======================
print("\n"+"="*70+"\n 7. CV FINAL\n"+"="*70)
m_l = lgb.LGBMClassifier(**lgbm_p, random_state=SEED, verbose=-1, n_jobs=-1)
m_x = xgb.XGBClassifier(**xgb_p, random_state=SEED, eval_metric='logloss',
                          verbosity=0, n_jobs=-1)
m_c = CatBoostClassifier(**cat_p, random_seed=SEED, verbose=0)
res = {}
for name, m in [('LGBM', m_l), ('XGB', m_x), ('CatBoost', m_c)]:
    s = cross_val_score(m, X, y, cv=cv, scoring='accuracy')
    res[name] = s.mean()
    print(f"  {name:12s} -> {s.mean():.4f} +/- {s.std():.4f}")
vt = VotingClassifier(estimators=[('l', m_l), ('x', m_x), ('c', m_c)],
                       voting='soft', n_jobs=-1)
sv = cross_val_score(vt, X, y, cv=cv, scoring='accuracy')
res['Ensemble'] = sv.mean()
print(f"  {'Ensemble':12s} -> {sv.mean():.4f} +/- {sv.std():.4f}")
best_n = max(res, key=res.get); best_cv = res[best_n]
print(f"\n  >>> MELHOR: {best_n} = {best_cv:.4f}")

# ====================== 8. SUBMISSÃO (BLEND MULTI-SEED) ======================
print("\n"+"="*70+"\n 8. SUBMISSÃO — Blend 3 modelos x 5 seeds\n"+"="*70)
proba = np.zeros(len(X_test)); nm = 0
for off in range(5):
    s = SEED + off * 7
    for cls, p, sk in [
        (lgb.LGBMClassifier, {**lgbm_p, 'verbose': -1, 'n_jobs': -1}, 'random_state'),
        (xgb.XGBClassifier, {**xgb_p, 'eval_metric': 'logloss', 'verbosity': 0,
                              'n_jobs': -1}, 'random_state'),
        (CatBoostClassifier, {**cat_p, 'verbose': 0}, 'random_seed')]:
        m = cls(**{**p, sk: s}); m.fit(X, y)
        proba += m.predict_proba(X_test)[:, 1]; nm += 1
preds = (proba / nm >= 0.5).astype(int)
sub = pd.DataFrame({'id_solicitante': test_ids, 'inadimplente': preds})
sub.to_csv('submissao_kaggle.csv', index=False)
print(f"  {nm} modelos blendados -> submissao_kaggle.csv")
print(f"  Distribuição: {dict(sub['inadimplente'].value_counts())}")

# ====================== 9. FEATURE IMPORTANCE ======================
print("\n"+"="*70+"\n 9. FEATURE IMPORTANCE\n"+"="*70)
fi_m = lgb.LGBMClassifier(**lgbm_p, random_state=SEED, verbose=-1)
fi_m.fit(X, y)
imp = pd.Series(fi_m.feature_importances_, index=feat_names).sort_values(ascending=False)
print("  Top 15:")
for i, (f, v) in enumerate(imp.head(15).items()):
    print(f"    {i+1:2d}. {f:50s} {v:5d}  {'█'*int(v/imp.max()*25)}")
fig, ax = plt.subplots(figsize=(10, 8))
t20 = imp.head(20)
ax.barh(range(len(t20)), t20.values, color='steelblue')
ax.set_yticks(range(len(t20))); ax.set_yticklabels(t20.index, fontsize=9)
ax.invert_yaxis(); ax.set_xlabel('Importância')
ax.set_title('Top 20 Features — LightGBM')
plt.tight_layout(); plt.savefig('feature_importance.png', dpi=150); plt.close()

print(f"\n{'='*70}")
print(f" RESUMO: CV={best_cv:.4f} ({best_n}), {len(feat_names)} features, blend {nm} modelos")
print(f"{'='*70}")
