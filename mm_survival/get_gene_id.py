#!/usr/bin/python3

from bs4 import BeautifulSoup
import requests
import html
import time
import argparse

'''
Made in a boring day by @danilotat. Enjoy
'''
def get_symbol(html_file, ensgid):
	soup = BeautifulSoup(html_file, 'html.parser')
	try:
		symbol_tag=soup.find('title').contents[0]
		symbol = symbol_tag.split(' ')[0]
		if symbol == ensgid:		
			return 'NaN'
		else:
			return symbol
	except AttributeError:
		print('Genecards is blocking your requests. Please try again later with less geneid')
		exit()


def gene_card_request(ensgid):
	''' 
	Passing Ensembl gene id, return its webpage on Genecards
	'''
	url_to_request='https://www.genecards.org/cgi-bin/carddisp.pl?gene=' + ensgid
	headers = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_10_1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/39.0.2171.95 Safari/537.36'}
	r = requests.get(url_to_request, headers=headers)
	gene_card_html = html.unescape(r.text)
	return gene_card_html
	
def get_entrez(html_file):
	''' 
	Given a valid Ensembl gene id, returns Entrez gene id
	'''
	soup = BeautifulSoup(html_file, 'html.parser')
	entrez_href = soup.find_all('a', {'title':'NCBI Entrez Gene'})
	# usually it will return redundant refer: we'll take just the first of them
	try:
		link = entrez_href[0]
		ncbi_url=link.get('href')
		entrez_id = ncbi_url.split('/')[-1]
		return entrez_id
	except IndexError:
		# empty list: no Entrez ID 
		return "NaN"

def fill_dict(gene_list):
	''' Fill a dict with Ensembl geneid, entrez, symbol
	'''
	conv_dict={}
	with open(gene_list, 'r') as engid_list:
		for line in engid_list:
			ensgid = line.rstrip()
			gcard_page = gene_card_request(ensgid)
			entrez_id = get_entrez(gcard_page)
			symbol = get_symbol(gcard_page, ensgid)
			conv_dict.setdefault(ensgid, []).extend([entrez_id,symbol])
			time.sleep(5)
	return conv_dict

if __name__ == '__main__':
	parser = argparse.ArgumentParser(description="Giving a list of valid Ensembl gene ID, query GeneCards to get Entrez and symbols.")
	parser.add_argument("--list", help="Input list of Ensembl gene ID, one per line.")
	parser.add_argument("--gene", help="Single gene mode")
	args = parser.parse_args()
	if (args.list == None and args.gene == None):
		parser.print_help()
		exit()
	if args.gene == None:
		res_dict = fill_dict(gene_list=args.input)
		for ensgid in res_dict.keys():
			print('{},{},{}'.format(ensgid,res_dict[ensgid][0], res_dict[ensgid][1]))
	else:
		gcard_page = gene_card_request(args.gene)
		print('{},{},{}'.format(args.gene,get_entrez(gcard_page), get_symbol(gcard_page, args.gene)))