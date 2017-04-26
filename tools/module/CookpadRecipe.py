# _*_ coding: utf-8 -*-

import MySQLdb 

class CookpadRecipe:
    """
    MySQLに登録したcookpad_dataを扱うためのクラス
    """
    def __init__(self, 
                 usr="fujino", 
                 password="", 
                 socket="/var/run/mysqld/mysqld.sock", 
                 host="localhost", 
                 db="cookpad_data"):
	print("passwd:",password)
        self.connection = MySQLdb.connect(host=host, 
                                db=db, 
                                user=usr, 
                                passwd=password, 
                                charset='utf8', 
                                unix_socket=socket)
        self.cursor = self.connection.cursor()


    def __del__(self):
        self.cursor.close()
        self.connection.close()


    def execute_sql(self, sql):
        """
            output: query result tuple(row1.tuple(col1, ... ,coln), row2.tuple(col1, ... , coln), ... ) 
        """
        self.cursor.execute(sql)
        return self.cursor.fetchall()


    def get_ingredients(self, recipe_id):
        """
            input: category_id
            output: a list of ingredients 
        """
        sql = "select name from ingredients where recipe_id='%s';" % recipe_id
        ingredients = self.execute_sql(sql)
        ingredients =  [x[0] for x in ingredients]
        return ingredients 


    def get_category_title(self, category_id):
        """
            output: a list of category ids
        """
        sql = "select title from search_categories where id = '%s';" % category_id
        title = self.execute_sql(sql)[0][0]
        return title 


    def get_category_ids(self):
        """
            output: a list of category ids
        """
        sql = "select id from search_categories;"
        ids = self.execute_sql(sql)
        ids =  [x[0] for x in ids]
        return ids 


    def get_children_category_id(self, category_id):
        """
            input: category_id
            output: a list of child category ids (not including its own)
        """
        sql = "select id from search_categories where parent_id = '%s';" % category_id
        children = self.execute_sql(sql)
        children =  [x[0] for x in children]
        return children


    def get_descendants_category_id(self, category_id):
        """
            input: category_id
            output: a list of descendants category ids (including its own)
        """
        descendants = []
        descendants.append(category_id)
        children = self.get_children_category_id(category_id)
        for child in children:
            descendants += self.get_descendants_category_id(child) #reccursion
        return descendants


    def get_recipe_ids_from_category(self, category_id):
        """
            input: title of category 
            output: a list of recipes contained in input category 
        """
        descendant_ids = self.get_descendants_category_id(category_id)
        sql = "select distinct recipe_id from search_category_recipes where search_category_id in (" 
        for descendant_id in descendant_ids:
            sql += "'%s'," % descendant_id 
        sql = sql[:-1] + ");" # remove comma
        recipes = self.execute_sql(sql)
        recipes =  [x[0] for x in recipes]
        return recipes

    def get_recipe_ids_from_ingredients(self, ingredients):
        """
            input: list of ingredients 
            output: a list of recipes contained in input ingredients 
        """
        sql = "select distinct recipe_id from ingredients where "
        sql += "name regexp '%s';" % '|'.join(ingredients)
        recipes = self.execute_sql(sql)
        recipes =  [x[0] for x in recipes]
        return recipes


if __name__ == "__main__":
    connection = MySQLdb.connect(
        host="localhost", 
        db="cookpad_data", 
        user="fujino", 
        passwd="", 
        charset="utf8", 
        unix_socket="/var/run/mysqld/mysqld.sock")
    cookpad_recipe = CookpadRecipe(connection)

#     recipe_id = '7a9d10ab5eb506a3e66b0e115a1ce84b0dfe7a39'
#     result = cookpad_recipe.get_ingredients(recipe_id)
#     print (result)

    title = u'野菜のおかず' 
    result = cookpad_recipe.get_descendants_category_id_from_title(title)
    for r in result: 
        sql = "select title from search_categories where id = '%s';" % r 
        t = cookpad_recipe.execute_sql(sql)
        print(t[0][0])
