import os
import random
import re
import unicodedata
import inflect
import emoji
import contractions
import nltk
import pickle
import IPython
import gensim
import pyLDAvis.gensim_models
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from gensim import corpora
import static

# Older libraries (gensim) needs explicit download
nltk.download('punkt')
nltk.download('stopwords')


# Predict a new document based on a saved model.
def predict_new_document(data):
    dictionary = gensim.corpora.Dictionary.load('data/dictionary.gensim')
    corpus = pickle.load(open('data/corpus.pkl', 'rb'))
    lda = gensim.models.ldamodel.LdaModel.load('data/model1.gensim')

    preprocess_step_1(False, data, "data/tmp_doc.csv")
    preprocess_step_2("data/tmp_doc.csv", "data/final_doc.csv")
    tokens = tokenize_new_document("data/final_doc.csv")
    # Flatten the list
    tokens = [x for xs in tokens for x in xs]
    # transform to BOW
    data_bow = dictionary.doc2bow(tokens)
    return lda.get_document_topics(data_bow)


# Function that opens the original data file or data and preprocesses it
def preprocess_step_1(isfile, fin, output):
    if isfile:
        data = open(fin, encoding="utf8")
    else:
        data = fin
    # Load the data file
    # Temporary output file required for the next step
    output = open(output, "w", encoding="utf8")

    # Perform preprocessing per line in the document
    for single_line in data:
        # Remove all URL's
        single_line = re.sub(
            r"(https?:\/\/)(\s)*(www\.)?(\s)*((\w|\s)+\.)*([\w\-\s]+\/)*([\w\-]+)((\?)?[\w\s]*=\s*[\w\%&]*)*",
            "", single_line)
        # Remove initials
        single_line = re.sub(r"(\~|\^|\*|\-|\/).[^\"\ \n]*", "", single_line)
        # Remove emojis
        emoji_pattern = re.compile("["
                                   u"\U0001F600-\U0001F64F"  # emoticons
                                   u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                                   u"\U0001F680-\U0001F6FF"  # transport & map symbols
                                   u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                                   u"\U0001F1F2-\U0001F1F4"  # Macau flag
                                   u"\U0001F1E6-\U0001F1FF"  # flags
                                   u"\U0001F600-\U0001F64F"
                                   u"\U00002702-\U000027B0"
                                   u"\U000024C2-\U0001F251"
                                   u"\U0001f900-\U0001f9ff"
                                   u"\U0001F1F2"
                                   u"\U0001F1F4"
                                   u"\U0001F620"
                                   u"\u200d"
                                   u"\u2640-\u2642"
                                   "]+", flags=re.UNICODE)
        single_line = emoji_pattern.sub(r'', single_line)
        # Remove broken or incomplete words
        single_line = re.sub(r"((\ {1}).[^\ ]*\.{3})", "", single_line)
        # Remove mentions with '@'
        single_line = re.sub(r"(\@.[^\ ]*)", "", single_line)
        # Write the processed line to the temporary file
        output.write(single_line)

    # Close files from the filesystem
    if isfile:
        data.close()
    output.close()
    print("The data has been preprocessed (1/2)")


# Separate processing to replace contractions
def preprocess_step_2(fin, output):
    # Load the temporary data file
    data = open(fin, encoding="utf8")
    # Final output file required for the next step
    output = open(output, "w", encoding="utf8")

    # Replace the contractions in every line
    # Note that doing this in the first preprocessing step results in incorrect words.
    # This needs to be executed after and separate from the other step.
    for single_line in data:
        output.write(contractions.fix(single_line))

    # Close files from the filesystem
    data.close()
    output.close()
    # Remove the temporary data file
    os.remove(fin)
    print("The data has been preprocessed (2/2)")


# Normalize a single line
def normalize(line):
    # Initialize engine to transform numbers to their textual representation
    p = inflect.engine()
    # The normalized line
    new_line = []
    # Tokenize the line by word using NLTK
    line = nltk.word_tokenize(line)

    # Construct a custom stopwords list which extends the one from NLTK stopwords.
    custom_stopwords = stopwords.words('english')
    custom_stopwords.extend(['https', 'ðy', 'us', 'dm', 'please', 'thanks', 'send', 'sent', 'thank', 'help',
                             'hi', 'amp', 'need', 'new', 'u', 'like', 'saying', 'go', 'number', 'look', 'well',
                             'see', 'im', 'a', 'help', 'tt', 'sorry', 'get', 'soon', 'would', 'follow', 'lets',
                             'details', 'nt', 'un', 'de', 'que', 'la', 'en', 'le', 'el', 'je', 'ca', 'e', 'x',
                             'gt', 'oh', 'let', 'link', 'else', 'got', 'fix', 'going', 'try', 'how', 'cc', 'hello',
                             'here', 'a', 'aa', 'ya', 'oa', 'ta', 'sure', 'hey', 'dona', 'ii ', 'ita', 'ðyt', 'cant',
                             'know', 'still', 'via', 'happy', 'trying', 'wed', 'thats', 'apologies', 'sorry', 'you',
                             'provide', 'personal', 'back', 'two', 'one', 'dtw', 'hear', 'glad', 'yes', 'no', 'using',
                            'time', 'anything', 'account', 'team', 'nice'])

    # Normalize every token in a line.
    for token in line:
        # Remove non ascii characters
        token = unicodedata.normalize('NFKD', token)
        # Make sure everything is in lowercase
        token = token.lower()
        # Remove punctuation
        token = re.sub(r'[^\w\s]', '', token)
        # Process numbers/digits
        if token.isdigit():
            try:
                token = p.number_to_words(token)
            except inflect.NumOutOfRangeError:
                # If a number cannot be transformed, it means that it's out of range and omitting is preferred.
                continue
        # Only include tokens that are larger than 2, else omit
        if len(token) <= 2:
            continue
        # Check that the token is not empty and is not a stopword
        if token != '' and token not in custom_stopwords:
            # Append the token to the new line
            new_line.append(token)
    return new_line


# Tokenize the final data
def tokenize(filein):
    # New tokenized data
    data = []
    # Open the file and normalize each line
    with open(filein, encoding="utf8") as fin:
        for line in fin:
            tokens = normalize(line)
            # Check if tokens is not empty and add the tokens to the final token list randomly
            if random.random() > .99 and tokens:
                data.append(tokens)

    print("The data has been tokenized.")
    return data


# Tokenize a new document
def tokenize_new_document(filein):
    # New tokenized data
    data = []
    # Open the file and normalize each line
    with open(filein, encoding="utf8") as fin:
        for line in fin:
            tokens = normalize(line)
            # Check if tokens is not empty and add the tokens to the final token list randomly
            if tokens:
                data.append(tokens)

    print("The data has been tokenized.")
    return data


# Prepare and analyze the data
def analyze(data):
    # Prepare the data for the models
    dictionary = corpora.Dictionary(data)
    corpus = [dictionary.doc2bow(text) for text in data]
    pickle.dump(corpus, open('data/corpus.pkl', 'wb'))
    dictionary.save('data/dictionary.gensim')
    number_of_topics = 10

    # Model 1: Latent Dirichlet Allocation
    model1 = gensim.models.ldamulticore.LdaModel(corpus, num_topics=number_of_topics, id2word=dictionary, passes=5)
    model1.save('data/model1.gensim')
    # Visualize the outcome with pyLDAvis
    vis = pyLDAvis.gensim_models.prepare(model1, corpus, dictionary=dictionary)
    pyLDAvis.save_html(vis, 'templates/LDA.html')
    # Load the topics and their words
    model1_topics = model1.print_topics(num_words=10)
    print("Model 1 (LDA): Top 10 topics")
    for topic in model1_topics:
        print(topic)
    print("Model 1 (LDA): Finished")

    # Model 2: Latent Semantic Indexing
    model2 = gensim.models.lsimodel.LsiModel(corpus, num_topics=number_of_topics, id2word=dictionary)
    model2.save('data/model2.gensim')
    # Load the topics and their words
    model2_topics = model2.print_topics(num_words=10)
    print("Model 2 (LSI): Top 10 topics")
    for topic in model2_topics:
        print(topic)
    print("Model 2 (LSI): Finished")


# Run this script locally to obtain the model and save cloud resources and thus money.
if __name__ == '__main__':
    preprocess_step_1(True, "data/data.csv", "data/tmp_data.csv")
    preprocess_step_2("data/tmp_data.csv", "data/final_data.csv")
    analyze(tokenize('data/final_data.csv'))
