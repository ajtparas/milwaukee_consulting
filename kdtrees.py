from sklearn.neighbors import KDTree
import pandas as pd



class Search:

    def __init__(self,df):
        df = df

    # function clean preprocesses columns by converting numerical
    # columns to categorical and provides levels for every value
    def clean(self, df):

        # create copy of original dataframe
        df_copy = df.copy()

        # convert column type to category
        df_copy['BLD_TYPE'] = df_copy['BLD_TYPE'].astype('category')
        # create levels of each value in column
        df_copy['BLD_TYPE'] = df_copy['BLD_TYPE'].cat.codes

        df_copy['APPRAISER'] = df_copy['APPRAISER'].astype('category')
        df_copy['APPRAISER'] = df_copy['APPRAISER'].cat.codes

        df_copy['NBHD'] = df_copy['NBHD'].astype('category')
        df_copy['NBHD'] = df_copy['NBHD'].cat.codes

        df_copy['QUAL'] = df_copy['QUAL'].astype('category')
        df_copy['QUAL'] = df_copy['QUAL'].cat.codes

        df_copy['COND'] = df_copy['COND'].astype('category')
        df_copy['COND'] = df_copy['COND'].cat.codes

        df_copy['KITCHEN_RATING'] = df_copy['KITCHEN_RATING'].astype('category')
        df_copy['KITCHEN_RATING'] = df_copy['KITCHEN_RATING'].cat.codes

        df_copy['FULL_BATH_RATING'] = df_copy['FULL_BATH_RATING'].astype('category')
        df_copy['FULL_BATH_RATING'] = df_copy['FULL_BATH_RATING'].cat.codes

        df_copy['HALF_BATH_RATING'] = df_copy['HALF_BATH_RATING'].astype('category')
        df_copy['HALF_BATH_RATING'] = df_copy['HALF_BATH_RATING'].cat.codes

        # create dataframe subset with those features we only want to test on
        df_subset = df_copy[
            ["BLD_TYPE", "APPRAISER", "NBHD", "QUAL", "COND", "KITCHEN_CT", "KITCHEN_RATING", "FULL_BATH_CT",
             "FULL_BATH_RATING", "HALF_BATH_CT",
             "HALF_BATH_RATING", "FINISHED_AREA", "LAND_SF"]]
        return df_subset

    # function neighbor takes the processed dataframe, the original data
    def neighbor(self, test_point, number_of_neighbors):

        # call on function clean to preprocess the dataset for searching
        cleaned_df = self.clean(df)

        # Creation of kd tree object
        tree = KDTree(cleaned_df)
        # find distances and indices of those closest neighbor from search test point
        dist, ind = tree.query(cleaned_df[test_point:test_point + 1], k=number_of_neighbors + 1)

        # window viewing options
        pd.set_option('max_columns', None)
        pd.set_option('expand_frame_repr', False)

        # print the test point
        print("Test Point:")
        print(df[test_point:test_point + 1])
        print();
        print()

        # create list for printing indices of closest neighbors
        houses = []
        for x in range(number_of_neighbors + 1):
            houses.append(x)
        houses.pop(0)

        # using ind variable from query, print the k nearest neighbors
        for i in houses:
            print("Most similar neighbor number: " + str(i))
            print(df[ind[0, i]:ind[0, i] + 1])
            print()

# import imputed dataset
df = pd.read_csv("imputeddatafinal.csv")
# subset for only houses with a sale price
df = df[df["SALE_PRICE"].notnull()]
# create Search object with input dataframe
house = Search(df)


# call neighbor search using first parameter test point and second parameter number of neighbors to call
house.neighbor(1, 5)




