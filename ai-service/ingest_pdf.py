import os
import uuid
from datetime import datetime
from pymongo import MongoClient
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv

# 1. Load Environment Variables
load_dotenv()
MONGO_URI = os.environ.get("MONGO_URI") 
if not MONGO_URI:
    print("❌ Error: MONGO_URI is missing from .env file!")
    exit(1)

# 2. Connect to MongoDB Atlas
mongo_client = MongoClient(MONGO_URI)
db = mongo_client["neofit"]
collection = db["knowledge_documents"]

# 3. Initialize embedding model
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

def ingest_pdf(file_name, sport="general"):
    # target the books directory
    file_path = os.path.join("books", file_name)
    print(f"🚀 Starting ingestion for: {file_path}")
    
    try:
        # load pdf content
        loader = PyPDFLoader(file_path)
        docs = loader.load()
        
        # split text into chunks
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        chunks = text_splitter.split_documents(docs)
        
        # skip if empty or scanned images
        if not chunks:
            print(f"⚠️ Warning: '{file_path}' is empty or contains only scanned images. Skipping...")
            return  
        
        # extract source filename for citations
        source_file = os.path.basename(chunks[0].metadata.get('source', 'Unknown Book'))

        documents_to_insert = []
        
        for chunk in chunks:
            page_num = chunk.metadata.get('page', 0) + 1 
            original_text = chunk.page_content
            
            # append citation to chunk
            enriched_content = f"[Source: {source_file}, Page: {page_num}]\n{original_text}"
            
            # generate vector embeddings
            vector = embeddings.embed_query(enriched_content)
            
            # Create MongoDB Document structure
            doc = {
                "_id": str(uuid.uuid4()), # توليد ID فريد
                "sport": sport,
                "topic": "training_knowledge",
                "content": enriched_content,
                "embedding": vector,
                "created_at": datetime.utcnow()
            }
            documents_to_insert.append(doc)
        
        # Insert all chunks into MongoDB at once for better performance
        if documents_to_insert:
            collection.insert_many(documents_to_insert)
            print(f"✅ Successfully ingested {len(chunks)} chunks with citations from {source_file}!")
        
    except Exception as e:
        print(f"❌ Error processing '{file_path}': {e}. Skipping to next book...")

if __name__ == "__main__":
    ingest_pdf("workout_guide.pdf", sport="general")
    #ingest_pdf("boxing_manual.pdf", sport="boxing")
    #ingest_pdf("Sport Nutrition.pdf", sport="nutrition")
    # ingest_pdf("Training For Warriors .pdf", sport="combat_fitness")
    #ingest_pdf("basics_of_strength_and_conditioning_manual.pdf", sport="strength")
    #ingest_pdf("b2264_Essentials_of_Strength_Training_and_Conditioning_4th_Edition-sport.ta4a.us.pdf", sport="strength")
    #ingest_pdf("feismo.com-strength-and-conditioning-for-sports-performance-pr_d00518cbe765de0ebc0ca2ec075c7957.pdf", sport="strength")
    # ingest_pdf("2264_Essentials_of_Strength_Training_and_Conditioning_4th_Edition-sport.ta4a.us.pdf", sport="strength")
    #ingest_pdf("Biomechanics of Punching—The Impact of Effective Mass and Force Transfer on Strike Performance.pdf", sport="boxing")
    #ingest_pdf("Effects_of_Plyometric_Training_on_Physical_Perform.pdf", sport="general")
    #ingest_pdf("biomechanicsoftheleadstraightpunchofdiffrentlevelboxers.pdf", sport="boxing")
    #ingest_pdf("Reciprocal Forearm Flexion-Extension Resistance Training Elicits Comparable Increases in Muscle Strength and Size With and Without Blood Flow Restriction.pdf", sport="general")
    # ingest_pdf("Special-Issue-Strength-and-conditioning-for-combat-sports-athletes-Revista-de-Artes-Marciales-Asiaticas.pdf", sport="combat_fitness")
    #ingest_pdf("Strength_and_Conditioning_for_Football.pdf", sport="football")
    #ingest_pdf("Strength_and_Conditioning_for_Soccer_Players.1.pdf", sport="football")
    #ingest_pdf("strength-and-conditioning-for-combat-sports-9781785004063_compress.pdf", sport="combat_fitness")
    #ingest_pdf("The-Physiology-of-Exercise-1774446321._print.pdf", sport="general")
    #ingest_pdf("tudor_bompa_carlo_buzzichelli-periodization_training_for_sports-human_kinetics__2015_.pdf", sport="general")