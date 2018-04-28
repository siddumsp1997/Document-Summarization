import nltk
import os
import numpy
import re
import sys
import math
from pythonrouge.pythonrouge import Pythonrouge

# Class to store sentence (node) information
class node_data:

	# A node contains sentence
	def __init__(self,str_sentence):

		self.sentence = str_sentence
		self.word_set = set()
		self.word_list = []
		self.tf = {}
		self.idf = {}		



# This text processing module is wrt to every folder of Topic 'i'
def Text_processing_module(i):


	folder_node_list = []

	#file_list = os.listdir("/home/sid/Downloads/Assignement2_IR/Topic"+str(i+1))
	file_list = os.listdir("./Topic"+str(i+1))

	temp_tkn = nltk.data.load('tokenizers/punkt/english.pickle')

	#iterate thru every file
	for file in file_list:

		#file_ob = open("/home/sid/Downloads/Assignement2_IR/Topic"+str(i+1)+"/"+file,"r")
		file_ob = open("./Topic"+str(i+1)+"/"+file,"r")

		# concatenating all the text in the folder to one entity
		file_text = file_ob.read()

		#print file 

		## text processing
		file_node_list = []

		#Either headline or headstart exists

		headline_start = file_text.find("<HEADLINE>")

		if headline_start == -1:

			header_start = file_text.find("<HEADER>")
			header_end = file_text.find("</HEADER>")

			temp_str1 = file_text[header_start+8:header_end]
			temp_str1 = temp_str1.strip()
			#print (" \n\n\n\HEADING :" + temp_str1)
			node = node_data(temp_str1)
			file_node_list.append(node)

		else:

			headline_end = file_text.find("</HEADLINE>")	
			temp_str2 = file_text[headline_start+10:headline_end]
			temp_str2 = temp_str2.strip()
			#print (" \n\n\nHEADING :" + temp_str2)
			node = node_data(temp_str2)
			file_node_list.append(node)

		index = 0

		#print file_node_list

		#Infinite while loop
		while True:

			para_start = file_text.find("<P>",index)
			para_end = file_text.find("</P>",para_start)

			if para_start != -1:

				pseudo_start = para_start + 3

				temp_sent = temp_tkn.tokenize(file_text[pseudo_start:para_end])
				
				for sent in temp_sent:

					#print sent

					temp_node = node_data(sent)

					file_node_list.append(temp_node)


			else:

				break

			index = para_end

		# End of infinite while loop

		#print file_node_list


		folder_node_list = folder_node_list + file_node_list

		# End of outer for loop ( for every file in folder of Topic 'i')


	return folder_node_list		

#End of function




# module to retrieve all the words (excluding stop words) and removing special characters
def getwordlist(node):

	sent = node.sentence
	sent = sent.lower()
	sent =re.sub("[^a-zA-Z]+"," ", sent)
	sent = sent.strip()
	word_list = sent.split(" ")

	stop_words = nltk.corpus.stopwords.words('english')

	#word_list1 = filter(lambda x: x not in stop_words, word_list)
	word_list1 = [x for x in word_list if x not in stop_words]

	word_list1 = filter(lambda x: x !='', word_list1)

	return word_list1

#end of function	





# Module to generate tf-idf vectors corresponding to the sentences
def generate_tf_idf_vectors(node_list):


	# Dictionary for storing the entire vocabulary
	# Vocabulary stores the no of nodes in which a 
	# particular word appears

	words_database = {}

	#Calculation of tf
	for node in node_list:

		node.word_list = getwordlist(node)
		#print node.word_list
		node.word_set = set(node.word_list)
		#print node.word_set

		
		for word in node.word_set:

			node.tf[word]  = 0

			if word not in words_database:

				words_database[word] = 1

			else:

				words_database[word] += 1

		#finding out the tf-vector of the node
		for word in node.word_list:

			node.tf[word] += 1	



	#Calculation of idf

	i = 0

	N = len(words_database)

	nodes_to_be_removed = []

	for node in node_list:

		for word in node.word_set:

			ni = words_database[word]
			#print "word = "+ word + "  N = "+ str(N)+ " ni = "+str(ni)
			node.idf[word] = math.log(N*1.0/ni)

		tmpsum = 0.0 

		if len(node.word_set) == 0:

			nodes_to_be_removed.append(i)

		else:

			for word in node.word_set:

				tmpsum += math.pow(node.tf[word]*node.idf[word],2)


				if tmpsum == 0.0:

					nodes_to_be_removed.append(i)


		i = i + 1		

	#print("Size of vocabulary : "+str(len(words_database))+"\n\n")


	#Removing invalid nodes (nodes containing invalid elements)

	final_node_list = []

	l = len(node_list)

	for i in range(0,l):

		if i not in nodes_to_be_removed:

			final_node_list.append(node_list[i])			


	return final_node_list	

# End of function




#Calculating idf-modified-cosine value between 2 nodes in a graph
def idf_modified_cosine_calc(node1, node2):


	numerator = 0.0
	denominator1 = 0.0
	denominator2 = 0.0

	for word in node1.word_set:

		tmp_term = node1.tf[word]*node1.idf[word]

		denominator1 += math.pow(tmp_term,2)


	denominator1 = math.sqrt(denominator1)	

	#print "den1 = " + str(denominator1) +"\n"


	for word in node2.word_set:

		tmp_term = node2.tf[word]*node2.idf[word]

		denominator2 += math.pow(tmp_term,2)


	denominator2 = math.sqrt(denominator2)

	#print "den2 = " + str(denominator2) +"\n"

	common_words = node1.word_set.intersection(node2.word_set)


	for word in common_words:

		#print str(node1.idf[word])+ " ----  "+str(node2.idf[word])
		idf_sqr = math.pow(node1.idf[word],2)

		numerator += node1.tf[word]*node2.tf[word]*idf_sqr


	#print "num = " + str(numerator) +"\n"

	if denominator1 == 0.0 or denominator2 == 0.0:
		return -1

	result = (numerator*1.0) / (denominator1*denominator2)

	return result

#End of function




#Function to create graph using the node list
def graph_construction(node_list):

	#Graph initialisation
	graph = {}

	list_length = len(node_list)

	for i in range(0,list_length):

		graph[i] = {}


	#Graph construction 
	for i in range(0,list_length):

		for j in range(i+1,list_length):

			idf_modified_cos = idf_modified_cosine_calc(node_list[i], node_list[j])

			#Adding edge
			if idf_modified_cos > 0.0 :

				graph[i][j] = idf_modified_cos
				graph[j][i] = idf_modified_cos


	# End of double for loop

	return graph

#end of function




#Function to threshold the graph
def thresholding(graph, threshold, n, node_list):


	for i in graph.keys():

		for j in graph[i].keys():

			if graph[i][j] < threshold:

				del graph[i][j]
				del graph[j][i]	

	
	#End of for loop


	return graph

#End of function	



#Module for summary generation (using degree centrality algorithm)
def summary_generation_degree_centrality(graph, threshold, node_list):


	final_sentences_index = []

	word_count = 0


	while True:

		highest_degree = 0
		highest_degree_node = -1

		for i in graph.keys():

			if len(graph[i].keys()) > highest_degree:	
				
				highest_degree = len(graph[i].keys())
				highest_degree_node = i		


		final_sentences_index.append(highest_degree_node)


		if highest_degree == 0 :
			break


		for j in graph[highest_degree_node].keys():

			del graph[highest_degree_node][j]
			del graph[j][highest_degree_node]

			for k in graph[j].keys():

				del graph[k][j]

			graph[j] = {}


		word_count += len(node_list[highest_degree_node].sentence.strip().split(" "))

		#print("Sentence i :  " + node_list[highest_degree_node].sentence + "\n")

		if word_count >= 250:

			break

	#End of the infinite while loop

	#print(" word count = "+ str(word_count)+"\n")
	#print( " No of sentences selected = "+str(len(final_sentences_index)))

	return final_sentences_index

# End of function	

		 




#main function
if __name__ == '__main__':  
	
	#text processing module for retrieving the text from the documents of the folder

	for index in range(0,5):

		# generates the total list of nodes in the folder "Topic-index"

		print "Topic "+str(index+1) 
		print "-------\n"

		folder_node_list = Text_processing_module(index)

		node_list = generate_tf_idf_vectors(folder_node_list)


		# List of thresholds
		thresholds = [0.1, 0.2, 0.3]
		#thresholds = [0.1]

		gold_summary_ob = open("./GroundTruth/Topic"+str(index+1)+".1","r")
		gold_summary_text = gold_summary_ob.read()

		reference_sentences = gold_summary_text.split("\n")


		#print reference_sentences 

		#print "\n\n"


		for threshold in thresholds:

			#threshold = 0.1
			print "Threshold = "+str(threshold)+"\n"

			graph = graph_construction(node_list)

			graph = thresholding(graph, threshold, len(node_list), node_list)


			final_sentence_index_list = summary_generation_degree_centrality(graph, threshold, node_list)

			#final_summary = ""

			hypotheses_sentences = []

			for i in final_sentence_index_list:

				final_sentence = node_list[i].sentence.split('\n')
				#final_sentence.replace('\n',' ')
				final_sentence1 = " ".join(final_sentence)
				final_sentence1 = final_sentence1.strip()
				#pred_sum_ob.write(final_sentence1 +"\n")
				#final_summary += final_sentence1 + "\n"
				hypotheses_sentences.append(final_sentence1)


			#print(final_summary)
			#print(hypotheses_sentences)

			ref = [[reference_sentences]]
			hyp = [hypotheses_sentences]


			rouge = Pythonrouge(summary_file_exist=False,
                    summary=hyp, reference=ref,
                    n_gram=2, ROUGE_SU4=False, ROUGE_L=True,
                    recall_only=False, stemming=True, stopwords=True,
                    word_level=True, length_limit=True, length=600,
                    use_cf=False, cf=95, scoring_formula='average',
                    resampling=True, samples=1000, favor=True, p=0.5)
			
			score = rouge.calc_score()

			print score
			print "\n\n"

		#End of thresholds for loop

	#End of for loop index -> (0,5)




#End of Main function		
	
