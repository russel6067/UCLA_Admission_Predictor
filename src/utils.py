import matplotlib.pyplot as plt
import seaborn as sns

def plot_actual_vs_predicted(y_test, predictions):
    fig, ax = plt.subplots()
    sns.scatterplot(x=y_test, y=predictions, ax=ax)
    ax.set_xlabel("Actual Admit Chance")
    ax.set_ylabel("Predicted Admit Chance")
    ax.set_title("Actual vs Predicted Admission Chances")
    return fig