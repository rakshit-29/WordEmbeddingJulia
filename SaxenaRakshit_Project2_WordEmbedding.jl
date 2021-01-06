using Pkg
using Embeddings
using Plots
using WordTokenizers
using TextAnalysis
using DelimitedFiles
using Distances
using Statistics
using MultivariateStats

embd=load_embeddings(Word2Vec)

# Loading Google's own Word2Vec pre-trained embeddings

const get_word_index = Dict(word=>ii for (ii,word) in enumerate(embd.vocab))
# Function for getting word index in the embedded table

function get_embedding(word)
    ind = get_word_index[word]
    emb = embd.embeddings[:,ind]
    return emb
end

#Function for getting the vectors for a specific word

get_embedding("human")

a=embd.vocab

#a shows the vocab of our embedded table

b=embd.embeddings

#b shows the  embeddings in our embedded table

size(a)

#size of a

size(b)

#size of b

get_word_index["human"]

#at what index our word is present

cosine(x,y)=1-cosine_dist(x,y)

#using distances.jl we derive a formula for cosine similarity 

cosine(get_embedding("dog"),get_embedding("puppy"))

#checking the similarity between dog and puppy

cosine(get_embedding("laptop"),get_embedding("medicine"))

cosine(get_embedding("cat"), get_embedding("kitten"))>cosine(get_embedding("song"),get_embedding("car"))

#returns true because cat and kitten are more similar than song and car are

function closest(v, n=25)
    list=[(x,cosine(embd.embeddings'[x,:], v)) for x in 1:size(embd.embeddings)[2]]
    topn_idx=sort(list, by = x -> x[2], rev=true)[1:n]
    return [embd.vocab[h] for (h,_) in topn_idx]
end

#function for finding similar words or closest words

closest(get_embedding("girl"))

closest(get_embedding("marine")+get_embedding("animal"))

#we can see whale, dolphin, manatee are all marine animals

closest(mean([get_embedding("day"),get_embedding("night")]))

#this shows what comes between day and night, afternoon, evening are some examples

closest(get_embedding("man")-get_embedding("woman")+get_embedding("queen"))

#what is man to x when woman is to queen, therefore x=king

closest(get_embedding("lion")-get_embedding("lioness")+get_embedding("tigress"))

#what is x to tigress when lion is to lioness, therefore x is tiger

closest(get_embedding("sentiment")+get_embedding("language"))

#as we can see pessimism, mood are there in our 25 element array

x1=get_embedding("boots");
y1=get_embedding("shoe");
using Plots
Plots.scatter(x1,y1,size=(180,180),color="purple")

#visualizing similarity between the vectors

x2=get_embedding("phone");
y2=get_embedding("cucumber");
using Plots
Plots.scatter(x2,y2,size=(180,180),color="purple")

# we can there are many outliers which presents that phone and cucumber are very dissimalar words

using CorpusLoaders

dataset_train_pos=CorpusLoaders.load(IMDB("train_pos"))
#loading train_pos set from the dataset

dataset_train_neg=CorpusLoaders.load(IMDB("train_neg"))
#loading train_neg set from the dataset

dataset_test_pos=CorpusLoaders.load(IMDB("test_pos"))
#loading test_pos set from the dataset

dataset_test_neg=CorpusLoaders.load(IMDB("test_neg"))
#loading test_neg set from the dataset

using Base.Iterators
ptrain=collect(take(dataset_train_pos,2))
#convert it into an array of array of strings

ntrain=collect(take(dataset_train_neg,2))
#convert it into an array of array of strings

ptest=collect(take(dataset_test_pos,2))
#convert it into an array of array of strings

ntest=collect(take(dataset_test_neg,2))
#convert it into an array of array of strings

using Unicode
function convert_clean_arr(arr)
    arr = string.(arr)
    arr = Unicode.normalize.(arr, stripmark=true)
    arr = map(x -> replace(x, r"[^a-zA-Z0-9_]" => " "), arr)
    return arr
end

#function to remove punctuation marks

ptr=string.(ptrain)
#converting to string

ntr=string.(ntrain)
#converting to string

pte=string.(ptest)
#converting to string

nte=string.(ntest)
#converting to string

ptr_pun=convert_clean_arr(ptr)
p1=string.(ptr_pun)
join(p1)
#converting from any array to array string then joining it to become one large string


ntr_pun=convert_clean_arr(ntr)
n1=string.(ntr_pun)
join(n1)
#converting from any array to array string then joining it to become one large string

pte_pun=convert_clean_arr(pte)
p2=string.(pte_pun)
join(p2)
#converting from any array to array string then joining it to become one large string

nte_pun=convert_clean_arr(nte)
n2=string.(nte_pun)
join(n2)
#converting from any array to array string then joining it to become one large string

using TextAnalysis: NaiveBayesClassifier, fit!, predict

#using naive bayes classifier for sentiment analysis between positive and negative

model1=NaiveBayesClassifier([:positive,:negative])

fit!(model1,join(p1),:positive)
#fitting train_pos model as positive

fit!(model1,join(n1),:negative)
#fitting train_neg model as negative

fit!(model1,join(p2),:positive)
#fitting test_pos model as positive

fit!(model1,join(n2),:negative)
#fitting test_neg model as negative

predict(model1,"it shows a tender side")
#prediction for the string comes as positive which is one hundred percent correct so accuracy is actually good

predict(model1,"unnatural feelings for a pig")
#prediction for the string comes as negative which is one hundred percent correct so accuracy is actually good

predict(model1,"The sign of a good movie is that it can toy with our emotions")
#prediction for the string comes as positive which is one hundred percent correct so accuracy is actually good

predict(model1,"once again Mr. Costner has dragged out a movie")
#prediction for the string comes as negative which is one hundred percent correct so accuracy is actually good

predict(model1,"very cocky overconfident ashton kutcher")
#prediction for the string comes as negative but the accuracy is less because 'ashton kutcher' is in positive models too


