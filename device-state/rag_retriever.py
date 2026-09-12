import os
import re

class ASHARetriever:
    def __init__(self, chunks_dir='/home/ubuntu/asha_knowledge/chunks'):
        self.chunks = []
        for fname in sorted(os.listdir(chunks_dir)):
            if fname.endswith('.txt'):
                with open(os.path.join(chunks_dir, fname), 'r', encoding='utf-8') as f:
                    self.chunks.append(f.read())
    
    def retrieve(self, query, top_k=3):
        '''Simple keyword-based retrieval: score chunks by word overlap.'''
        query_words = set(re.findall(r'\w+', query.lower()))
        scores = []
        for i, chunk in enumerate(self.chunks):
            chunk_words = set(re.findall(r'\w+', chunk.lower()))
            overlap = len(query_words & chunk_words)
            scores.append((overlap, i, chunk))
        
        # Sort by overlap descending, take top_k
        scores.sort(reverse=True, key=lambda x: x[0])
        top_chunks = [chunk for _, _, chunk in scores[:top_k] if _ > 0]
        return top_chunks

if __name__ == '__main__':
    # Test retrieval
    r = ASHARetriever()
    query = "baby not getting milk breastfeeding"
    results = r.retrieve(query, top_k=2)
    print(f'Query: {query}')
    print(f'Retrieved {len(results)} chunks')
    for i, chunk in enumerate(results):
        print(f'\\nChunk {i+1}: {chunk[:200]}...')
