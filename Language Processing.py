# Case Study 2: Language Processing

# Counting Words ---------------------------------------------------------------------------------------------
text = "This is my text. We're keeping this text short to keep things manageable."

def count_words(text):
    # Count the number of times each word occurs in text (str). Return dictionary where keys are unique words and values are word counts.
    word_counts = {}
    for word in text.split(" "): # split words by space
        # known word
        if word in word_counts:
            word_counts[word] += 1
        # unknown word
        else:
            word_counts[word] = 1
    return word_counts
    
count_words(text)


# skip punctuation
def count_words(text):
    # Count the number of times each word occurs in text (str). Return dictionary where keys are unique words and values are word counts.
    # Skip punctuation.
    
    # how to deal with punctuation and capitalize 
    text = text.lower() 
    # define characters that we will be skeeping and convert them into spaces
    skips = [".", ",", ";", ":", "'", '"'] 
    for ch in skips:
        text = text.replace(ch, "")
    
    word_counts = {}
    for word in text.split(" "): # split words by space
        # known word
        if word in word_counts:
            word_counts[word] += 1
        # unknown word
        else:
            word_counts[word] = 1
    return word_counts
    
count_words(text)

# integrated python function
from collections import Counter

def count_words_fast(text):
    # Count the number of times each word occurs in text (str). Return dictionary where keys are unique words and values are word counts.
    # Skip punctuation.
    
    # how to deal with punctuation and capitalize 
    text = text.lower() 
    # define characters that we will be skeeping and convert them into spaces
    skips = [".", ",", ";", ":", "'", '"'] 
    for ch in skips:
        text = text.replace(ch, "")
        
    word_counts = Counter(text.split(" ")) # replace for loopt with Counter function from collections library
    return word_counts
    
count_words_fast(text)

# both methods return identical values
count_words(text) == count_words_fast(text)
count_words(text) is count_words_fast(text) # but different objects


# Reading in a Book -------------------------------------------------------------------------------------------------------------
def read_book(title_path):
    # Read a book and return it as a string
    
    with open(title_path, "r", encoding = "UTF-8") as current_file: # "r" is opening the file for reading
        text = current_file.read()
        text.replace("\n", "").replace("\r", "") # replace breaks 
    return text
    
# read romeo and juliet
text = read_book("./Books/English/shakespeare/Romeo and Juliet.txt")

# find a line
ind = text.find("What's in a name?")

# sample text
sample_text = text[ind: ind + 1000]


# Computing Word Frequency Statistics --------------------------------------------------------------------------------------
# return frequencies of each word
def word_stat(word_counts): # word_counts is from previous function
    # Return number of unique words and word frequencies.
    
    num_unique = len(word_counts) 
    counts = word_counts.values() # frequencies of each word in our text
    return (num_unique, counts)
    
text = read_book("./Books/English/shakespeare/Romeo and Juliet.txt")
word_counts = count_words(text)
(num_unique, counts) = word_stat(word_counts)
print(num_unique)
sum(counts) # total words

# compare with german translation
text = read_book("./Books/German/shakespeare/Romeo und Julia.txt")
word_counts = count_words(text)
(num_unique, counts) = word_stat(word_counts)
print(num_unique)
sum(counts) 


# Reading Multiple Files ----------------------------------------------------------------------------------------------
# how to navigate directories. Read every book from a directory s
import os
book_dir = "./Books"
os.listdir(book_dir) # how many items in the directory, returns a list

# read all books in the directory
for language in os.listdir(book_dir): # oops over books
    for author in os.listdir(book_dir + "/" + language): # loops over authors
        for title in os.listdir(book_dir + "/" + language + "/" + author): # loops over titles
            inputfile = book_dir + "/" + language + "/" + author + "/" + title
            print(inputfile)
            text = read_book(inputfile)
            (num_unique, counts) = word_stat(count_words(text))

# PANDAS ----------------------------------------------------------------------------------------------------------------
# useful for panel data. Provides additional data structures and data analysis functionalities for python.
# specially useful for manipulating numerical tables and time series data.
import pandas as pd

# dataframe identical to the one in R
table = pd.DataFrame(columns = ("name", "age"))

table.loc[1] = "James", 22 # first row
table.loc[1] = "Jess", 32 # second row

table
table.columns

# how to use pandas to save our previous results
import pandas as pd
stas = pd.DataFrame(columns=("language", "author", "title", "len", "unique"))
title_num = 1

for language in os.listdir(book_dir): # oops over books
    for author in os.listdir(book_dir + "/" + language): # loops over authors
        for title in os.listdir(book_dir + "/" + language + "/" + author): # loops over titles
            inputfile = book_dir + "/" + language + "/" + author + "/" + title
            print(inputfile)
            text = read_book(inputfile)
            (num_unique, counts) = word_stat(count_words(text))
            stats.loc[title_num] = language, author, title, sum(counts), num_unique
            title_num += 1
            
stats.head() # top 5         
stats.tail() # last 5

# capitalize author and delete .txt from title name
import pandas as pd
stas = pd.DataFrame(columns=("language", "author", "title", "len", "unique"))
title_num = 1

for language in os.listdir(book_dir): # oops over books
    for author in os.listdir(book_dir + "/" + language): # loops over authors
        for title in os.listdir(book_dir + "/" + language + "/" + author): # loops over titles
            inputfile = book_dir + "/" + language + "/" + author + "/" + title
            print(inputfile)
            text = read_book(inputfile)
            (num_unique, counts) = word_stat(count_words(text))
            stats.loc[title_num] = language, author.capitalize(), title.replace(".txt", ""), sum(counts), num_unique
            title_num += 1
            
# Reading Multiple Files: Question 2  ---
# What pandas method allows you to insert a row to a dataframe?
pd.loc()

# Plotting Book Statistics--------------------------------------------------------------------------------------
stats.length # access a column, as in R changin $ by a dot (.)
stats.unique
stats["length"] # equivalent to access column
stats["unique"]

import matplotlib.pyplot as plt

# Simple scatter plor
plt.plot(stats.length, stats.unique, "bo")

# loglog version
plt.loglog(stats.length, stats.unique, "bo")

# stratify data
stats[stats.lenguage == "English"] # as in R
stats[stats.lenguage == "French"]

plt.figure(figsize = (10, 10))
subset = stats[stats.lenguage == "English"] 
plt.loglog(subset.length, subset.unique, "o", label = "English", color = "crimson")
subset = stats[stats.lenguage == "French"] 
plt.loglog(subset.length, subset.unique, "o", label = "French", color = "forestgreen")
subset = stats[stats.lenguage == "German"] 
plt.loglog(subset.length, subset.unique, "o", label = "German", color = "orange")
subset = stats[stats.lenguage == "Portuguese"] 
plt.loglog(subset.length, subset.unique, "o", label = "Portuguese", color = "blueviolet")
plt.legend()
plt.xlabel("Book length")
plt.ylabel("Number of unique words")
plt.savefig("lang_plot.pdf")