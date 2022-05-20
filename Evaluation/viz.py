
# comprehensive_score
import matplotlib.pyplot as plt

def viz_compreh (df_scores_p):
    with open(df_scores_p, 'r') as f:
        score_df = pd.read_csv(f)
    fig = plt.figure(figsize=(9,6))
    y = [score for score in score_df['compre_score']]
    x = [str(i) for i in range(1,11)]
    plt.plot(x,y, 'v--', color= 'darkorange' )#, label= query )

    plt.xlabel('subsets')
    plt.ylabel('comprehensiveness scores')
    plt.legend(loc='upper left', ncol = 1)
    # plt.title( 'comprehensive scores over 10 subsets')
    plt.show()
    fig.savefig(r"C:\Users\huyen\OneDrive\Documents\GitHub\CORD-19-KG\Evaluation\result\KG_eval\comprehensiveness_10folds.png" )


def viz_QA_performance(result_df_p, metric):
    ###not done. Here:http://localhost:8888/notebooks/OneDrive/Documents/GitHub/CORD-19-KG/Evaluation/QA_evaluation.ipynb
    with open(df_scores_p, 'r') as f:
        result_df = pd.read_csv(f)
    # Set color for each model
    colors = {'query_1': 'lightcoral', 'query_2': 'darkorange', 'query_3': 'lime', 'query_4': 'steelblue',
              'query_5': 'purple'}
    # Set marker for each model
    markers = {'query_1': '1--', 'query_2': 'v--', 'query_3': '^--', 'query_4': '*--', 'query_5': 'o--'}

    fig = plt.figure(figsize=(9, 6))
    if metric == 'precision':
        key = 0
    elif metric == 'recall':
        key = 1
    elif metric == 'F1':
        key = 2
    else:
        raise ValueError('Error: wrong evaluation metric given!')
    for query, fold_scores in by_queries.items():
        y = [scores[key] for scores in fold_scores]
        x = [i for i in range(1, 11)]
        plt.plot(x, y, markers[query], color=colors[query], label=query)

    plt.ylabel(metric)
    plt.xlabel('subsets')
    plt.legend(loc='upper left', ncol=2)
    plt.title('%s scores over 10 subsets' % metric)
    plt.show()
    fig.savefig(r"C:\Users\hn0139\Documents\GitHub\CORD-19-KG\Evaluation\result\%s.png" % metric)


viz_performance('C:\Users\huyen\OneDrive\Documents\GitHub\CORD-19-KG\Evaluation\result\QA_eval\QA_eval_scores.csv', F1')
