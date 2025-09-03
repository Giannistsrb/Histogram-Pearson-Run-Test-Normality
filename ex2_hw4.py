import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import chi2
from scipy.stats import norm

#========================================================================
#CHI SQUARE TEST:
def CHI2_TEST(PDF, PDF_ERRORS, theoritical_expectation):

    e_PDF               = np.array(PDF_ERRORS)
    CHI_SQUARE_TEST_PDF = np.sum(((PDF - theoritical_expectation) ** 2) / e_PDF ** 2 )
    p_value             = chi2.sf(CHI_SQUARE_TEST_PDF, len(bin_centers))

    return CHI_SQUARE_TEST_PDF, p_value

#RUN TEST:
def RUN_TEST(PDF, theoritical_expectation):

    difference = np.array(PDF-theoritical_expectation)

    #THE NUMBER OF SIGN CHANGES IN DIFFERENCES:
    R = 1  # Initialization of the number of sign changes in differences
    for i in range(len(bin_edges) - 1): 
        if difference[i + 1] > 0 and difference[i] < 0:
            R = R + 1
        if difference[i + 1] < 0 and difference[i] > 0: 
            R = R + 1
    
    n1 = len(difference[difference > 0]) # The number of positive differences
    n2 = len(difference[difference < 0]) # The number of negative differences
    
    mean_R  = 2 * n1 * n2 / (n1 + n2) + 1
    sigma_R = np.sqrt((2 * n1 * n2 * (2 * n1 * n2 - n1 - n2)) / (((n1 + n2) ** 2) * (n1 + n2 - 1)))
    
    Z       = (R - mean_R) / sigma_R 

    return Z

#========================================================================

bin_edges   = [i / 2         for i in range(-6, 6)]
bin_centers = [i / 2 + 1 / 4 for i in range(-6, 6)]
bin_width   = 5e-1

alpha = 0.1 # Level of significance
mean  = 0   # Mean of theoritical normal distribution
sigma = 1   # Sigma of theoritical normal distribution

#The theoritical normal distribution expectation for every bin:
theoritical_expectation =  bin_width * norm.pdf(np.array(bin_centers), mean, sigma)

PDF1        = [0.0042, 0.0186, 0.0422, 0.0926, 0.1456, 0.1886, 
               0.1866, 0.1514, 0.0894, 0.0546, 0.0176, 0.0066]

PDF2        = [0.0056, 0.0214, 0.0422, 0.0844, 0.1296, 0.1660,
               0.1736, 0.1536, 0.1072, 0.0626, 0.0340, 0.0136]

PDF1_ERRORS = [0.0009, 0.0019, 0.0029, 0.0043, 0.0054, 0.0061,
               0.0061, 0.0055, 0.0042, 0.0033, 0.0019, 0.0011]

PDF2_ERRORS = [0.0011, 0.0021, 0.0029, 0.0041, 0.0051, 0.0058,
               0.0059, 0.0055, 0.0046, 0.0035, 0.0026, 0.0016]

#PDF1 histogram:
plt.figure(figsize = (8, 6))
plt.bar(x = bin_centers, height = PDF1, color = "yellow", edgecolor = "black", label = "PDF 1")
plt.errorbar(bin_centers, PDF1, yerr=PDF1_ERRORS, fmt='none', ecolor='black', capsize=7, 
             label = "PDF 1 - Errors")

#Theoritical expectation for PDF1:
x = np.linspace(min(bin_edges), max(bin_edges), 1000)
plt.plot(x, bin_width * norm.pdf(np.array(x), mean, sigma), color = "black", 
         label = "Theoritical Normal Distribution N(0,1)")
plt.plot(bin_centers, bin_width * norm.pdf(np.array(bin_centers), mean, sigma), 'o', 
         label = "Theoritical Points", color = "red")
plt.legend()
plt.title("Probability Density Function 1 ", fontsize=18)
plt.xlabel("Bins", fontsize=18)
plt.ylabel("Density", fontsize=18)
plt.grid(True)
plt.tight_layout()
plt.show()

#PDF2 histogram:
plt.figure(figsize = (8, 6))
plt.bar(x = bin_centers, height = PDF2, color = "yellow", 
        edgecolor = "black", label = "PDF 2")
plt.errorbar(bin_centers, PDF2, yerr=PDF2_ERRORS, fmt='none', ecolor='black', 
             capsize=7, label = "PDF 2 - Errors")

#Theoritical expectation for PDF2:
x = np.linspace(min(bin_edges), max(bin_edges), 1000) 
plt.plot(x, bin_width * norm.pdf(np.array(x), mean, sigma), color = "black", 
         label = "Theoritical Normal Distribution N(0,1)")
plt.plot(bin_centers, bin_width * norm.pdf(np.array(bin_centers), mean, sigma), 'o', 
         label = "Theoritical Points", color = "red")
plt.legend()
plt.title("Probability Density Function 2 ", fontsize=18)
plt.xlabel("Bins", fontsize=18)
plt.ylabel("Density", fontsize=18)
plt.grid(True)
plt.tight_layout()
plt.show()

#Chi2 test for PDF1:
CHI_SQUARE_TEST_PDF1, p_value_PDF1 = CHI2_TEST(PDF1, PDF1_ERRORS, theoritical_expectation)

#Chi2 test for PDF2:
CHI_SQUARE_TEST_PDF2, p_value_PDF2 = CHI2_TEST(PDF2, PDF2_ERRORS, theoritical_expectation)

#RUN test for PDF1:
RUN_TEST_PDF1 = RUN_TEST(PDF1, theoritical_expectation)

#RUN test for PDF2:
RUN_TEST_PDF2 = RUN_TEST(PDF2, theoritical_expectation)

#Print of the results:
print("=========CHI2 TEST===========")
print(f"For PDF1: Chi2 = {CHI_SQUARE_TEST_PDF1} and p-value = {p_value_PDF1}")
print(f"For PDF2: Chi2 = {CHI_SQUARE_TEST_PDF2} and p-value = {p_value_PDF2}")

print("=========RUN TEST===========")
print(f"For PDF1: Z = {RUN_TEST_PDF1}")
print(f"For PDF2: Z = {RUN_TEST_PDF2}")

# Check if results are statistically significant:
if p_value_PDF1 <= alpha:
    print("For PDF1 using the Chi2 test, we reject the null hypothesis. The data do not follow the theoretical normal distribution N(0,1).")
else:
    print("For PDF1 using the Chi2 test, we do not reject the null hypothesis. The data follow the theoretical normal distribution N(0,1).")
if p_value_PDF2 <= alpha:
    print("For PDF2 using the Chi2 test, we reject the null hypothesis. The data do not follow the theoretical normal distribution N(0,1).")
else:
    print("For PDF2 using the Chi2 test, we do not reject the null hypothesis. The data follow the theoretical normal distribution N(0,1).")
if RUN_TEST_PDF1 <= alpha:
    print("For PDF1 using the RUN test, we reject the null hypothesis. The data do not follow the theoretical normal distribution  N(0,1).")
else:
    print("For PDF1 using the RUN test, we do not reject the null hypothesis. The data follow the theoretical normal distribution  N(0,1).")
if RUN_TEST_PDF2 <= alpha:
    print("For PDF2 using the RUN test, we reject the null hypothesis. The data do not follow the theoretical normal distribution  N(0,1).")
else:
    print("For PDF2 using the RUN test, we do not reject the null hypothesis. The data follow the theoretical normal distribution  N(0,1).")





