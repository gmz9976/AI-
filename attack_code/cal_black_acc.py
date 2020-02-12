

#152
#114
#96
avg_mse = 	17.92757
haven_mse = 3.848775045559988

a = (avg_mse-haven_mse) * 5.0 * 120.0
b = 128.0 - haven_mse

print("the black model has acc is {}".format(a/b))

