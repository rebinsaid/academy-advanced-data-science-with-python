import matplotlib.pyplot as plt

coefs = pd.DataFrame(pipeline_lr['model'].coef_[0], index=features, columns=['Importances'])
coefs_plotter = (
    coefs
    .assign(negative = (coefs['Importances']<0).map({True: 'darkorange', False: 'royalblue'})) # bonus
    .assign(Importances = coefs['Importances'].abs())
    .sort_values('Importances')
)
coefs_plotter

fig, ax = plt.subplots(figsize=(15,10))

coefs_plotter.plot(
    y='Importances', 
    kind='barh', 
    ax=ax, 
    legend=False,
    color=coefs_plotter['negative'] # bonus
)
ax.grid(True, axis='x')