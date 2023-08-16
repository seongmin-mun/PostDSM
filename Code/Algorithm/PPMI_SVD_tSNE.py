class PPMI_SVD_tSNE_Algorithm:

    def __init__ (self, window):
        self.window = window

    def PPMI_SVD_tSNE_Calculation(self):


        from collections import Counter
        import itertools
        import nltk
        from nltk.corpus import stopwords
        import numpy as np
        import pandas as pd
        from scipy import sparse
        from scipy.sparse import linalg
        from sklearn.preprocessing import normalize
        from sklearn.metrics.pairwise import cosine_similarity

        trainDir = "../../Data/Input/Fold/0Fold/Lo_train_0.csv"


        df = pd.read_csv(trainDir)
        print(df.head())
        headlines = df['Sentence'].tolist()
        headlines = [[tok for tok in headline.split()] for headline in headlines]

        headlines = [hl for hl in headlines if len(hl) > 1]

        print(headlines[0:20])

        tok2indx = dict()
        unigram_counts = Counter()
        for ii, headline in enumerate(headlines):
            if ii % 200000 == 0:
                print(f'finished {ii / len(headlines):.2%} of headlines')
            for token in headline:
                unigram_counts[token] += 1
                if token not in tok2indx:
                    tok2indx[token] = len(tok2indx)
        indx2tok = {indx: tok for tok, indx in tok2indx.items()}
        print('done')
        print('vocabulary size: {}'.format(len(unigram_counts)))
        print('most common: {}'.format(unigram_counts.most_common(10)))

        wordType = len(unigram_counts);

        for j in range(1, self.window):

            back_window = j
            front_window = j
            skipgram_counts = Counter()
            for iheadline, headline in enumerate(headlines):
                for ifw, fw in enumerate(headline):
                    icw_min = max(0, ifw - back_window)
                    icw_max = min(len(headline) - 1, ifw + front_window)
                    icws = [ii for ii in range(icw_min, icw_max + 1) if ii != ifw]
                    for icw in icws:
                        skipgram = (headline[ifw], headline[icw])
                        skipgram_counts[skipgram] += 1
                if iheadline % 200000 == 0:
                    print(f'finished {iheadline / len(headlines):.2%} of headlines')
            print('done')
            print('number of skipgrams: {}'.format(len(skipgram_counts)))
            print('most common: {}'.format(skipgram_counts.most_common(10)))

            row_indxs = []
            col_indxs = []
            dat_values = []
            ii = 0
            for (tok1, tok2), sg_count in skipgram_counts.items():
                ii += 1
                if ii % 1000000 == 0:
                    print(f'finished {ii / len(skipgram_counts):.2%} of skipgrams')
                tok1_indx = tok2indx[tok1]
                tok2_indx = tok2indx[tok2]

                row_indxs.append(tok1_indx)
                col_indxs.append(tok2_indx)
                dat_values.append(sg_count)

            wwcnt_mat = sparse.csr_matrix((dat_values, (row_indxs, col_indxs)))
            print('done')

            wwcnt_norm_mat = normalize(wwcnt_mat, norm='l2', axis=1)

            def ww_sim(word, mat, topn=len(tok2indx)):
                """Calculate topn most similar words to word"""
                indx = tok2indx[word]
                if isinstance(mat, sparse.csr_matrix):
                    v1 = mat.getrow(indx)
                else:
                    v1 = mat[indx:indx + 1, :]
                sims = cosine_similarity(mat, v1).flatten()
                sindxs = np.argsort(-sims)
                sim_word_scores = [(indx2tok[sindx], sims[sindx]) for sindx in sindxs[0:topn]]
                return sim_word_scores

            num_skipgrams = wwcnt_mat.sum()
            assert (sum(skipgram_counts.values()) == num_skipgrams)

            row_indxs = []
            col_indxs = []

            pmi_dat_values = []
            ppmi_dat_values = []
            spmi_dat_values = []
            sppmi_dat_values = []

            alpha = 0.75
            nca_denom = np.sum(np.array(wwcnt_mat.sum(axis=0)).flatten() ** alpha)
            sum_over_words = np.array(wwcnt_mat.sum(axis=0)).flatten()
            sum_over_words_alpha = sum_over_words ** alpha
            sum_over_contexts = np.array(wwcnt_mat.sum(axis=1)).flatten()

            ii = 0
            for (tok1, tok2), sg_count in skipgram_counts.items():
                ii += 1
                if ii % 1000000 == 0:
                    print(f'finished {ii / len(skipgram_counts):.2%} of skipgrams')
                tok1_indx = tok2indx[tok1]
                tok2_indx = tok2indx[tok2]

                nwc = sg_count
                Pwc = nwc / num_skipgrams
                nw = sum_over_contexts[tok1_indx]
                Pw = nw / num_skipgrams
                nc = sum_over_words[tok2_indx]
                Pc = nc / num_skipgrams

                nca = sum_over_words_alpha[tok2_indx]
                Pca = nca / nca_denom

                pmi = np.log2(Pwc / (Pw * Pc))
                ppmi = max(pmi, 0)

                spmi = np.log2(Pwc / (Pw * Pca))
                sppmi = max(spmi, 0)

                row_indxs.append(tok1_indx)
                col_indxs.append(tok2_indx)
                pmi_dat_values.append(pmi)
                ppmi_dat_values.append(ppmi)
                spmi_dat_values.append(spmi)
                sppmi_dat_values.append(sppmi)

            pmi_mat = sparse.csr_matrix((pmi_dat_values, (row_indxs, col_indxs)))
            ppmi_mat = sparse.csr_matrix((ppmi_dat_values, (row_indxs, col_indxs)))
            spmi_mat = sparse.csr_matrix((spmi_dat_values, (row_indxs, col_indxs)))
            sppmi_mat = sparse.csr_matrix((sppmi_dat_values, (row_indxs, col_indxs)))

            print('done')

            matrix_use = ppmi_mat

            if wordType < 500:
                embedding_size = wordType - 1
            else:
                embedding_size = 500

            uu, ss, vv = linalg.svds(matrix_use, embedding_size)
            print('vocab size: {}'.format(len(unigram_counts)))
            print('embedding size: {}'.format(embedding_size))
            print('uu.shape: {}'.format(uu.shape))
            print('ss.shape: {}'.format(ss.shape))
            print('vv.shape: {}'.format(vv.shape))

            unorm = uu / np.sqrt(np.sum(uu * uu, axis=1, keepdims=True))
            vnorm = vv / np.sqrt(np.sum(vv * vv, axis=0, keepdims=True))

            word_vecs = uu + vv.T
            word_vecs_norm = word_vecs / np.sqrt(np.sum(word_vecs * word_vecs, axis=1, keepdims=True))

            print(word_vecs_norm)

            from sklearn.manifold import TSNE
            X_embedded = TSNE(n_components=2, random_state=0).fit_transform(word_vecs_norm)

            wordList = []
            wordnum = 0
            for typeeach in indx2tok:
                wordList.append(indx2tok[wordnum])
                wordnum += 1

            tsne_df = pd.DataFrame({'X': X_embedded[:, 0], 'Y': X_embedded[:, 1], 'Word': wordList})
            tsne_df.to_csv("../../Data/Output/PPMI_SVD/Lo/t-SNE/Lo_tSNE_" + str(j) + ".csv")


            TSNE_dic = {}

            typenum = 0
            for typeeach in indx2tok:
                TSNE_dic[indx2tok[typenum]] = [X_embedded[typenum][0], X_embedded[typenum][1]]

                typenum = typenum + 1



            functionLo = ["FNS", "INS", "DIR", "EFF", "CRT", "LOC"]


            for function in functionLo:
                word = "(으)로/JKB" + "_" + function

                from numpy import dot
                from numpy.linalg import norm
                import numpy as np
                def cos_sim(A, B):
                    return dot(A, B) / (norm(A) * norm(B))


                target = np.array(TSNE_dic[word])
                outDir = "../../Data/Output/PPMI_SVD/Lo/Similarity/Lo_" + function + "_Similarity_" + str(j) + ".csv"

                f = open(outDir, 'w')
                tsnenum = 0
                for typeeach in indx2tok:
                    if indx2tok[tsnenum] != "Lo":
                        source = np.array(TSNE_dic[indx2tok[tsnenum]])

                        normal_sim = (cos_sim(target, source) + 1) / 2

                        data = str(indx2tok[tsnenum]) + "," + str(normal_sim)
                        f.write(data + "\n")
                    tsnenum = tsnenum + 1
                f.close()



