from pymongo import MongoClient

# Replace with your MongoDB Atlas connection string
connection_string = "mongodb+srv://Str_2364353:mjo2h5KxnbV5EMJG@cluster0.yvnzvii.mongodb.net/?retryWrites=true&w=majority"

try:
    client = MongoClient(connection_string)
    db = client.test_database  # Replace with your database name
    print("Connected to MongoDB Atlas successfully")
except Exception as e:
    print("Error connecting to MongoDB Atlas:", e)
