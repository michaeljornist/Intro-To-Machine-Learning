#################################
# Your name: Michael Jornist
#################################

import numpy as np
import matplotlib.pyplot as plt
from intervals import find_best_interval



class Assignment2(object):
    """Assignment 2 skeleton.

    Please use these function signatures for this assignment and submit this file, together with the intervals.py.
    """

    def sample_from_D(self, m):
        """Sample m data samples from D.
        Input: m - an integer, the size of the data sample.

        Returns: np.ndarray of shape (m,2) :
                A two dimensional array of size m that contains the pairs where drawn from the distribution P.
        """
        # x distributed uniformly on the interval [0,1] 
        x = np.random.uniform(0, 1, m)

        # Generate the corresponding y values based on the given probability distribution.
        y = np.zeros(m, dtype=int)
        
        for i in range(m):
            if 0 <= x[i] <= 0.2 or 0.4 <= x[i] <= 0.6 or 0.8 <= x[i] <= 1:
                y[i] = np.random.binomial(1, 0.8)
            else:
                y[i] = np.random.binomial(1, 0.1)

        # Combine the x and y arrays into a single 2D array.
        data = np.c_[x, y]

        return data


    def experiment_m_range_erm(self, m_first, m_last, step, k, T):
        """Runs the ERM algorithm.
        Calculates the empirical error and the true error.
        Plots the average empirical and true errors.
        Input: m_first - an integer, the smallest size of the data sample in the range.
               m_last - an integer, the largest size of the data sample in the range.
               step - an integer, the difference between the size of m in each loop.
               k - an integer, the maximum number of intervals.
               T - an integer, the number of times the experiment is performed.

        Returns: np.ndarray of shape (n_steps,2).
            A two dimensional array that contains the average empirical error
            and the average true error for each m in the range accordingly.
        """
        empirical_errors = []
        true_errors = []

        for m in range(m_first, m_last + 1, step):
            curr_emp_err = []
            curr_true_err = []

            for _ in range(T):
                sample = self.sample_from_D(m)
                sample = sample[np.argsort(sample[:, 0])] #Sorting the sample
                x, y = sample[:, 0], sample[:, 1]
                 # Assume find_best_interval is a function that finds the best intervals
                intervals, err_count = find_best_interval(x,y, k)
                emp_error = err_count/x.size
                # print(emp_error)
                true_error = self.calc_true_error(intervals)
                curr_emp_err.append(emp_error)
                curr_true_err.append(true_error)
            
            empirical_errors.append(np.mean(curr_emp_err))
            true_errors.append(np.mean(curr_true_err))

        m_values = range(m_first, m_last + 1, step)
        # Plotting the errors
        plt.figure(figsize=(10, 6))
        plt.plot(m_values, empirical_errors, label='Empirical Error')
        plt.plot(m_values, true_errors, label='True Error')
        plt.xlabel('Sample Size (m)')
        plt.ylabel('Error')
        plt.title('Empirical and True Errors as a Function of Sample Size')
        plt.legend()
        plt.grid(True)
        plt.show()

        return np.array(empirical_errors), np.array(true_errors)

    def experiment_k_range_erm(self, m, k_first, k_last, step):
        """Finds the best hypothesis for k= 1,2,...,10.
        Plots the empirical and true errors as a function of k.
        Input: m - an integer, the size of the data sample.
               k_first - an integer, the maximum number of intervals in the first experiment.
               k_last - an integer, the maximum number of intervals in the last experiment.
               step - an integer, the difference between the size of k in each experiment.

        Returns: The best k value (an integer) according to the ERM algorithm.
        """
        empirical_errors = []
        true_errors = []
        best_k = -1
        smallest_emp_err = 1
        sample = self.sample_from_D(m)
        sample = sample[np.argsort(sample[:, 0])] #Sorting the sample
        x, y = sample[:, 0], sample[:, 1]
        for k in range(k_first, k_last + 1, step):
            
            intervals, err_count = find_best_interval(x,y, k)
            emp_error = err_count/x.size
            if emp_error < smallest_emp_err:
                best_k = k
                smallest_emp_err = emp_error
            # print(emp_error)
            true_error = self.calc_true_error(intervals)
            empirical_errors.append(emp_error)
            true_errors.append(true_error)
            

        k_values = range(k_first, k_last + 1, step)
        # Plotting the errors
        plt.figure(figsize=(10, 6))
        plt.plot(k_values, empirical_errors, label='Empirical Error')
        plt.plot(k_values, true_errors, label='True Error')
        plt.xlabel('k')
        plt.ylabel('Error')
        plt.title('Empirical and True Errors as a Function of k')
        plt.legend()
        plt.grid(True)
        plt.show()

        return best_k

    def experiment_k_range_srm(self, m, k_first, k_last, step):
        """Run the experiment in (c).
        Plots additionally the penalty for the best ERM hypothesis.
        and the sum of penalty and empirical error.
        Input: m - an integer, the size of the data sample.
               k_first - an integer, the maximum number of intervals in the first experiment.
               k_last - an integer, the maximum number of intervals in the last experiment.
               step - an integer, the difference between the size of k in each experiment.

        Returns: The best k value (an integer) according to the SRM algorithm.
        """
        penaltys = []
        penalty_and_emp_errors = []            
        empirical_errors = []       
        true_errors = []
        best_k = -1
        smallest_emp_err_with_panlty = 1
        sample = self.sample_from_D(m)
        sample = sample[np.argsort(sample[:, 0])] #Sorting the sample
        x, y = sample[:, 0], sample[:, 1]
        for k in range(k_first, k_last + 1, step):
            
            penalty = 2 * np.sqrt((2 * k + np.log((2 * k**2) / 0.1)) / m)
            penaltys.append(penalty)
            intervals, err_count = find_best_interval(x,y, k)
            emp_error = err_count/x.size
            penalty_and_emp_errors.append(penalty+emp_error)
            if penalty+emp_error < smallest_emp_err_with_panlty:
                best_k = k
                smallest_emp_err_with_panlty = emp_error
            # print(emp_error)
            true_error = self.calc_true_error(intervals)
            empirical_errors.append(emp_error)
            true_errors.append(true_error)
            

        k_values = range(k_first, k_last + 1, step)
        # Plotting the errors
        plt.figure(figsize=(10, 6))
        plt.plot(k_values, empirical_errors, label='Empirical Error')
        plt.plot(k_values, true_errors, label='True Error')
        plt.plot(k_values, penaltys, label='Penalty')
        plt.plot(k_values, penalty_and_emp_errors, label='Penalty + Empirical Error')
        plt.xlabel('k')
        plt.ylabel('Error')
        plt.title('Empirical,True Errors,Penelty and Empirical eeror with penelty as a Function of k')
        plt.legend()
        plt.grid(True)
        plt.show()

        return best_k

    def cross_validation(self, m):
        """Finds a k that gives a good test error.
        Input: m - an integer, the size of the data sample.

        Returns: The best k value (an integer) found by the cross validation algorithm.
        """
        sample = self.sample_from_D(m)
        split_index = int(0.8 * m)
        train_sample = sample[:split_index]
        val_sample = sample[split_index:]
        train_sample = train_sample[np.argsort(train_sample[:, 0])]
        val_sample = val_sample[np.argsort(val_sample[:, 0])]
        x_train, y_train = train_sample[:, 0], train_sample[:, 1]
        x_val, y_val = val_sample[:, 0], val_sample[:, 1]
        smallest_emp_err_for_val = 1
        best_k = -1
        for k in range(1,11):
            intervals, err_count = find_best_interval(x_train, y_train, k)
            err_count_for_val = 0

            for i in range(x_val.size):
                if self.predict(x_val[i],intervals) != y_val[i]:
                    err_count_for_val += 1
            
            emp_err_for_val = err_count_for_val/x_val.size
            if emp_err_for_val < smallest_emp_err_for_val:
                smallest_emp_err_for_val = emp_err_for_val
                best_k = k
        return best_k
    



    #################################
    # Recevies a list of tuples represent a list of intervals I and return
    # the true error of the hypnosis of I
    def calc_true_error(self, intervals):
        I = [(0, 0.2), (0.4, 0.6), (0.8, 1)]
        J = intervals
        I_comp = self.calc_complement(I)   
        J_comp = self.calc_complement(J)

        I_inter_J = self.calc_intersection(I,J)
        I_inter_J_comp = self.calc_intersection(I,J_comp) 
        I_comp_inter_J = self.calc_intersection(I_comp,J)
        I_comp_inter_J_comp = self.calc_intersection(I_comp,J_comp)

        prop_to_I_inter_J = self.calc_intervals_length(I_inter_J) #P(X in I inter J)
        prop_to_I_inter_J_comp = self.calc_intervals_length(I_inter_J_comp) #P(X in I inter J comp)
        prop_to_I_comp_inter_J = self.calc_intervals_length(I_comp_inter_J) #P(X in I comp inter J)
        prop_to_I_comp_inter_J_comp = self.calc_intervals_length(I_comp_inter_J_comp) #P(X in I comp inter J comp)
        return 0.2*prop_to_I_inter_J + 0.8*prop_to_I_inter_J_comp + 0.9*prop_to_I_comp_inter_J + 0.1*prop_to_I_comp_inter_J_comp
        
    
    def predict(self, x, intervals):
        """Predicts the label for a given x based on the intervals."""
        for start, end in intervals:
            if start <= x <= end:
                return 1
        return 0

    
    def calc_complement(self,I):
        complement = []
        current_start = 0

        for interval in I:
            l, u = interval
            if current_start < l:
                complement.append((current_start, l))
            current_start = u

        if current_start < 1:
            complement.append((current_start, 1))

        return complement

    def calc_intersection(self,I, J):
        intersection = []

        for i_start, i_end in I:
            for j_start, j_end in J:
                if i_end > j_start and j_end > i_start:  # Check for overlap
                    intersection_start = max(i_start, j_start)
                    intersection_end = min(i_end, j_end)
                    if intersection_start < intersection_end:
                        intersection.append((intersection_start, intersection_end))

        return intersection

    def calc_intervals_length(self ,I):
        total =0
        for start,end in I:
            total += end - start
        return total
    
    #################################


if __name__ == '__main__':
    print("loading")
    ass = Assignment2()
    ass.experiment_m_range_erm(10, 100, 5, 3, 100)
    ass.experiment_k_range_erm(1500, 1, 10, 1)
    ass.experiment_k_range_srm(1500, 1, 10, 1)
    ass.cross_validation(1500)
    
    print("done")