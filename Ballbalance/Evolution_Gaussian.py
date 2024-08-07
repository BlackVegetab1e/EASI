import numpy as np
import matplotlib.pyplot as plt
import time



class Evo_Gaussian():

    def __init__(self,DNA_SIZE,DNA_BOUND,POP_SIZE,SURVIVE_RATE, need_clip = None):
        self.DNA_SIZE = DNA_SIZE
        self.DNA_BOUND = DNA_BOUND
        self.POP_SIZE = POP_SIZE
        self.SURVIVE_RATE = SURVIVE_RATE
        self.ELITE_SIZE = int(POP_SIZE*SURVIVE_RATE)
        self.need_clip = need_clip



        self.pop = np.empty((self.POP_SIZE, self.DNA_SIZE))

        for i in range(self.DNA_SIZE):
            self.pop[:,i] = DNA_BOUND[i][0]+(np.random.random(self.POP_SIZE))*(DNA_BOUND[i][1]-DNA_BOUND[i][0])
        
        self.elite = self.pop[:self.ELITE_SIZE]
        self.mean_vec = np.zeros(self.DNA_SIZE)
        self.var = np.zeros(self.DNA_SIZE)


        weights = (np.log(0.5 * (self.POP_SIZE + 1)) -
                        np.log(np.arange(1, self.POP_SIZE + 1)))

        self.norm_pos_weights = (weights[:self.ELITE_SIZE] /
                            sum(weights[:self.ELITE_SIZE]))
        

    def calculate_weighted_mean_var(self, datas, weight):
        '''
        datas为m*n维数组,其中m表示共有m组数据,n表示一个数据中包含n个元素
        weight是衡量每条数据重要性的指标,是一个m*1维数组
        '''
        weighted_mean = np.empty(self.DNA_SIZE)
        weighted_var  = np.empty(self.DNA_SIZE)
        
        weighted_mean = weight.dot(datas)
        weighted_var = weight.dot((datas-weighted_mean)**2)

        return weighted_mean, weighted_var


    def make_kids_Gaussian(self):


        # 这里添加一个考虑权重的参数分布

        self.mean_vec, self.var = self.calculate_weighted_mean_var(self.elite, self.norm_pos_weights)



        # self.mean_vec = self.elite.mean(axis=0)
        
        # self.var = self.elite.var(axis=0)
        # print(self.mean_vec)
        # print(self.var)
        # print(2*self.var)
        cov_mat = np.diag(self.var)

        
        pop = np.random.multivariate_normal(self.mean_vec,
                                              cov_mat,
                                              size=self.POP_SIZE)

        for i in range(self.DNA_SIZE):
            if self.need_clip is None:
                pop[:,i] = np.clip(pop[:,i], self.DNA_BOUND[i][0], self.DNA_BOUND[i][1])
            else:
                if self.need_clip[i]:
                    pop[:,i] = np.clip(pop[:,i], self.DNA_BOUND[i][0], self.DNA_BOUND[i][1])
                else:
                    pop[:,i] = np.clip(pop[:,i], 0, 1e10)

        self.pop = pop
        return pop
    

    def select_elite(self, reward):
        pop_matrix = np.c_[reward,self.pop]
        haha = reward[(-pop_matrix[:, 0]).argsort()]

        pop_matrix = pop_matrix[(-pop_matrix[:, 0]).argsort()]  # 这里需要升序排列，所以加负号

        self.elite = pop_matrix[:self.ELITE_SIZE, -self.DNA_SIZE:]
        
    def show_plot_mean_var(self, gen, search_logdir):
        
        print('param mean@gen',gen,':',self.mean_vec)
        print('param var@gen',gen,':',self.var)

        np.savetxt( search_logdir+str(gen)+"_mean.csv", self.mean_vec, delimiter="," )
        np.savetxt( search_logdir+str(gen)+"_var.csv", self.var, delimiter="," )
        # mean_values = self.mean_vec
        # variance_values = self.var
        # # 绘制图表
        # labels = range(11)
        # x = np.arange(len(labels))

        # plt.bar(x - 0.2, mean_values, width=0.4, label='Mean', align='center')
        # plt.bar(x + 0.2, variance_values, width=0.4, label='Variance', align='center')

        # plt.xlabel('Variables')
        # plt.ylabel('Values')
        # plt.title('Mean and Variance of Variables')
        # plt.xticks(x, labels)
        # plt.legend()

        # plt.show()


    
    def get_pop(self):
        return self.pop


