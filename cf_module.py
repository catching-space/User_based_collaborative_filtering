from scipy import spatial

class collaborative_filtering:
    
    def __init__(self, df):
        self.df = df
        self.sm_df = None
        self.similarity_func = None
        self.ms_df = None
        self.pred_df = None
        self.recommend_df = None
        self.algorithm = None
        
        
    def __euclidian_similarity(self, vector_1, vector_2):
        idx = np.array(vector_1).nonzero()[0]
        vector_1 = np.array(vector_1)[idx]
        vector_2 = np.array(vector_2)[idx]
        
        idx = np.array(vector_2).nonzero()[0]
        vector_1 = np.array(vector_1)[idx]
        vector_2 = np.array(vector_2)[idx]
        
        return np.linalg.norm(vector_1 - vector_2)
    
    
    def __cosine_similarity(self, vector_1, vector_2):
        idx = np.array(vector_1).nonzero()[0]
        vector_1 = np.array(vector_1)[idx]
        vector_2 = np.array(vector_2)[idx]
        
        idx = np.array(vector_2).nonzero()[0]
        vector_1 = np.array(vector_1)[idx]
        vector_2 = np.array(vector_2)[idx]
        
        return 1 - spatial.distance.cosine(vector_1, vector_2)
    
    
    def __assign_similarity_func(self, similarity_func):
        if similarity_func == "euclidian_similarity":
            self.similarity_func = self.__euclidian_similarity
        elif similarity_func == "cosine_similarity":
            self.similarity_func = self.__cosine_similarity
            
    
    def similarity_matrix(self, similarity_func="cosine_similarity"):
        """
        Calculates the similarity between users.
        """
        self.__assign_similarity_func(similarity_func)
        index = self.df.index

        matrix = []
        for idx_1, value_1 in self.df.iterrows():
            row = []
            for idx_2, value_2 in self.df.iterrows():
                row.append(self.similarity_func(value_1, value_2))
            matrix.append(row)
            
        return pd.DataFrame(matrix, columns=index, index=index)
    
    
    def mean_score(self, sm_df, target, closer_count):
        """
        It estimates the similarity(preference) score of the target user by considering closer_count.
        """
        self.sm_df = sm_df
        self.ms_df = self.sm_df.drop(target)
        self.ms_df = self.ms_df.sort_values(target, ascending=False)
        self.ms_df = self.ms_df[:closer_count]
        self.ms_df = self.df.loc[self.ms_df.index]
        
        self.pred_df = pd.DataFrame(columns=self.df.columns)
        self.pred_df.loc["user"] = self.df.loc[target]
        self.pred_df.loc["mean"] = self.ms_df.mean()
        
        return self.pred_df
    
    
    def recommend(self, pred_df):
        """
        Recommend the appropriate article to the target user.
        """
        self.recommend_df = self.pred_df.T
        self.recommend_df = self.recommend_df[self.recommend_df["user"] == 0]
        self.recommend_df = self.recommend_df.sort_values("mean", ascending=False)
        
        return list(self.recommend_df.index)
    
    
    def run(self, similarity_func, target, closer_count):
        """
        Recommend articles at once using similarity_func, target, and closer_count.
        """
        self.__assign_similarity_func(similarity_func)

        self.sm_df = self.similarity_matrix(self.similarity_func)
        self.pred_df = self.mean_score(self.sm_df, target, closer_count)
        
        return self.recommend(self.pred_df)
    
    
    def __mse(self, value, pred): 
        idx = value.nonzero()[0]
        value, pred = np.array(value)[idx], np.array(pred)[idx]
        idx = pred.nonzero()[0]
        value, pred = np.array(value)[idx], np.array(pred)[idx]
        
        return sum((value - pred) ** 2) / len(idx)
    
    
    def __rmse(self, value, pred):
        idx = value.nonzero()[0]
        value, pred = np.array(value)[idx], np.array(pred)[idx]
        idx = pred.nonzero()[0]
        value, pred = np.array(value)[idx], np.array(pred)[idx]
        
        return np.sqrt(sum((value - pred) ** 2) / len(idx))
    
    
    def __mae(self, value, pred):
        idx = value.nonzero()[0]
        value, pred = np.array(value)[idx], np.array(pred)[idx]
        idx = pred.nonzero()[0]
        value, pred = np.array(value)[idx], np.array(pred)[idx]
        
        return sum(np.absolute(value - pred)) / len(idx)
    
    
    def __assign_algorithm(self, algorithm):
        if algorithm.lower() == "mse":
            self.algorithm = self.__mse
        elif algorithm.lower() == "rmse":
            self.algorithm = self.__rmse
        elif algorithm.lower() == "mae":
            self.algorithm = self.__mae
            
    
    def evaluate(self, sm_df, target, closer_count, algorithm="mae"):
        """
        Measure the performance of the result using sm_df, target, closer_count, and algorithm.
        """
        self.__assign_algorithm(algorithm)
        users = self.df.index
        evaluate_list = []

        for target in users:
            pred_df = self.mean_score(sm_df, target, closer_count)
            evaluate_var = self.algorithm(self.pred_df.loc["user"],
                                          self.pred_df.loc["mean"])
            evaluate_list.append(evaluate_var)
            
        return np.average(evaluate_list)