import nltk # https://www.nltk.org
import pickle # https://docs.python.org/3/library/pickle.html
import os # https://docs.python.org/3/library/os.html
import nltk # https://www.nltk.org
import sys # https://docs.python.org/pt-br/3/library/sys.html
from math import log

# O etiquetador é responsável pelo processo de definição da classe gramatical 
# das palavras, de acordo com as funções sintáticas.
def criar_etiquetador():
    if os.path.isfile('mac_morpho.pkl'):
        # Carregando um modelo treinado
        input = open('mac_morpho.pkl', 'rb')
        tagger = pickle.load(input)
        input.close()
    else:
        # Obtendo as sentencas etiquetadas do corpus mac_morpho
        tagged_sents = nltk.corpus.mac_morpho.tagged_sents()
        
        # Instanciando etiquetador e treinando com as sentenças etiquetadas
        tagger = nltk.UnigramTagger(tagged_sents)
        # t2 = nltk.BigramTagger(tagged_sents, backoff=t1)    
        # tagger = nltk.TrigramTagger(tagged_sents, backoff=t2)

        # Salvar um modelo treinado em um arquivo para mais tarde usa-lo.
        output = open('mac_morpho.pkl', 'wb')
        pickle.dump(tagger, output, -1) 
        output.close()
    return tagger

def validar_argumentos():
    # É necessário passar como argumento o caminho para as bases e o caminho para
    # a consulta
    if len(sys.argv) <= 2:
        print('Número de argumentos inválido')
        exit()

    # receber  um argumento como  entrada  pela linha  de comando. Este argumento 
    # especifica o caminho de um arquivo texto que contém os caminhos de todos os
    # arquivos que compõem a base, cada um em uma linha.
    (base, consulta) = (sys.argv[1], sys.argv[2])

    # se caminho nao é arquivo ou nao existe o programa fecha
    if not os.path.isfile(base) or not os.path.isfile(consulta):
        print('Caminho para base inválido.')
        exit()
    
    return (base, consulta)

# Valida e retorna os argumentos da linha de comando
(caminhoBase, caminhoConsulta) = validar_argumentos()

# abre a base informada e le seus documentos
base = open(caminhoBase, 'r')
documentos = base.read().split('\n')
base.close()

# pega o diretorio da base
pastas = caminhoBase.split('/')
diretorio = ''.join(pastas[0:len(pastas)-1]) + '/' if len(pastas)-1 != 0 else ''

# Assuma que nesses  arquivos  texto, palavras  são separadas por um  ou mais
# dos seguintes caracteres: espaço em branco ( ), ponto (.), reticências(...)
# vírgula (,), exclamação (!), interrogação (?) ou enter (\n).
pontuacao = [',', '!', '?', '\n']

# PREP = preposição, ART = artigo, KC = conjunção coordenativa, KS = conjução
# subordinativo
retirarClassificacao = ['PREP', 'ART', 'KC', 'KS']

# entendimento do significado de um documento.
stopwords = nltk.corpus.stopwords.words('portuguese')

etiquetador = criar_etiquetador()

# A processing interface for removing morphological  affixes from words. This 
# process is known as stemming.
# https://www.nltk.org/api/nltk.stem.html
stemmer = nltk.stem.RSLPStemmer()

indicesInvertidos = {}
for numeroArquivo, documento in enumerate(documentos):
    # abre e le o conteudo de todos os documentos
    file = open(diretorio + documento)
    doc = file.read()
    file.close()

    # caracteres '.', ',', '!', '?', '...' e '\n' não devem ser considerados.
    # se o caracter for . troca por um espaço
    semPontuacao = [p if p != '.' else ' ' for p in doc if p not in pontuacao]
    palavras = ''.join(semPontuacao).split(' ')

    # stopwords não devem ser levadas em conta na geração do índice invertido
    semStopwords = [p for p in palavras if p not in stopwords]
    semEspacos = [p for p in semStopwords if p not in ' ']

    # classificação gramatical
    etiquetados = etiquetador.tag(semEspacos)

    # sem as classificações de preposição, conjunção e artigo
    semClassificacoes = [p[0] for p in etiquetados if p[1] not in retirarClassificacao]
    
    # extrair os radicais das palavras para o índice invertido
    radicais = [stemmer.stem(p) for p in semClassificacoes]
    
    # faz um indice invertido para o documento n
    indice = {p:(numeroArquivo + 1, radicais.count(p)) for p in radicais}

    # junta todos os indices em um só  indice invertido. Queria  muito fazer
    # com compreesão, mas fui incapaz :C
    for chave, valor in indice.items():
        if chave not in indicesInvertidos.keys():
            indicesInvertidos[chave] = {}
        indicesInvertidos[chave][valor[0]] = valor[1]

# ordena o indice invertido pela chave
indice = dict(sorted(indicesInvertidos.items()))

# escrev em arquivo o indice invertido
# escreve o indice invertido no arquivo indice.txt
arquivo = open('indice.txt', 'w')
for chave, valor in indice.items():
    arquivo.write(f'{chave}:')
    for k, v in valor.items():
        arquivo.write(f' {k},{v}')
    arquivo.write('\n')
arquivo.close()

# número de documentos na base
N = numeroArquivo+1

# calcula o tf 
tf = lambda f: 1 + log(f, 10) if f >= 1 else 0

# valores dos idf para cada termo
idf = {}

pesosDocumentos = {}

# calcula o peso dos termos pra cada documento
for i in range(1, N+1):
    pesosDocumentos[i] = {}
    for k in indice:
        idf[k] = log(N/len(indice[k]), 10)
        if i in indice[k]:
            pesosDocumentos[i][k] = tf(indice[k][i])*idf[k]
        else:
            continue

# escreve o arquivo de pesos
arquivo = open('pesos.txt', 'w')
for chave, valor in pesosDocumentos.items():
    arquivo.write(f'{documentos[int(chave)-1]}:')
    for k, v in valor.items():
        if v > 0.0:
            arquivo.write(f' {k},{v}')
    arquivo.write('\n')
arquivo.close()

# abre e le o conteudo do arquivo de consulta
consulta = open(caminhoConsulta, 'r')
termos = consulta.read().replace(' ', '').replace('\n', '').split('&')
consulta.close()

# extrair os radicais dos termos da consulta
radicais = [stemmer.stem(t) for t in termos]


# calcula o peso do termo da consulta
peso_consulta = lambda t: tf(radicais.count(t))*(idf[t] if t in idf else 0)

# peso de todos os termos da consulta
pesoConsulta = {t:peso_consulta(t) for t in radicais}

def calc_numerador(wi):
    valor = 0
    for termo, peso in pesoConsulta.items():
        if termo in wi:
            valor += wi[termo] * peso
    return valor

def calc_denominador(w):
    pesos = [p**2 for p in w.values()]
    return sum(pesos)**(1/2)

# calcula a similaridade dos documentos com a consulta
denominadorConsulta = calc_denominador(pesoConsulta)
similaridade = {}
for i in range(1, N+1):
    denominador = calc_denominador(pesosDocumentos[i]) * denominadorConsulta
    numerador = calc_numerador(pesosDocumentos[i]) 
    if denominador > 0:
        simi = (numerador / denominador)
        if simi >= 0.001: 
            similaridade[i] = simi

# ordena os resultados de forma descrente, do maior para o menor
similaridade = dict(sorted(similaridade.items(),key=lambda item: item[1] ,reverse=True))

# escreve arquivo das consultas
arquivo = open('resposta.txt', 'w')
arquivo.write(f'{len(similaridade)}\n')
for chave, valor in similaridade.items():
    arquivo.write(f'{documentos[chave-1]}: {valor}\n')
arquivo.close()