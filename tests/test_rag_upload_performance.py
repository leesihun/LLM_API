"""
Performance Testing Script for RAG Upload Optimizations
Compare original vs optimized upload methods
"""
import time
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import config
from tools.rag import RAGTool


def test_upload_performance(pdf_path: str, collection_name: str = "test_performance"):
    """
    Test and compare upload performance
    
    Args:
        pdf_path: Path to PDF file
        collection_name: Collection name for test
    """
    pdf_path = Path(pdf_path)
    
    if not pdf_path.exists():
        print(f"Error: PDF file not found: {pdf_path}")
        return
    
    if not pdf_path.suffix.lower() == '.pdf':
        print(f"Error: File is not a PDF: {pdf_path}")
        return
    
    # Get PDF page count
    try:
        import fitz
        doc = fitz.open(str(pdf_path))
        total_pages = len(doc)
        doc.close()
    except Exception as e:
        print(f"Warning: Could not get page count: {e}")
        total_pages = "unknown"
    
    print("=" * 80)
    print("RAG Upload Performance Test")
    print("=" * 80)
    print(f"PDF: {pdf_path.name}")
    print(f"Pages: {total_pages}")
    print(f"Size: {pdf_path.stat().st_size / 1024 / 1024:.1f} MB")
    print()
    
    username = "test_user"
    tool = RAGTool(username=username)
    
    # Test 1: Optimized upload (default)
    print("-" * 80)
    print("Test 1: Optimized Upload (PyMuPDF + Parallel)")
    print("-" * 80)
    
    start_time = time.time()
    
    try:
        result = tool.upload_document(
            collection_name=f"{collection_name}_optimized",
            document_path=str(pdf_path),
            use_optimized=True
        )
        
        optimized_time = time.time() - start_time
        
        if result["success"]:
            print("\n✓ Success!")
            print(f"  Upload time: {optimized_time:.1f}s")
            print(f"  Chunks created: {result['chunks_created']:,}")
            print(f"  Total chunks in collection: {result['total_chunks']:,}")
            
            if "timing" in result:
                print("\n  Timing breakdown:")
                for key, value in result["timing"].items():
                    if key != "total":
                        print(f"    {key.capitalize()}: {value:.1f}s ({value/optimized_time*100:.1f}%)")
        else:
            print(f"\n✗ Failed: {result.get('error', 'Unknown error')}")
            optimized_time = None
            
    except Exception as e:
        print(f"\n✗ Error: {e}")
        optimized_time = None
    
    print()
    
    # Test 2: Original upload (for comparison)
    print("-" * 80)
    print("Test 2: Original Upload (PyPDFLoader + Sequential)")
    print("-" * 80)
    print("Note: This may take significantly longer...")
    print()
    
    start_time = time.time()
    
    try:
        result = tool.upload_document(
            collection_name=f"{collection_name}_original",
            document_path=str(pdf_path),
            use_optimized=False
        )
        
        original_time = time.time() - start_time
        
        if result["success"]:
            print("\n✓ Success!")
            print(f"  Upload time: {original_time:.1f}s")
            print(f"  Chunks created: {result['chunks_created']:,}")
            print(f"  Total chunks in collection: {result['total_chunks']:,}")
        else:
            print(f"\n✗ Failed: {result.get('error', 'Unknown error')}")
            original_time = None
            
    except Exception as e:
        print(f"\n✗ Error: {e}")
        original_time = None
    
    print()
    
    # Comparison
    print("=" * 80)
    print("Performance Comparison")
    print("=" * 80)
    
    if optimized_time and original_time:
        speedup = original_time / optimized_time
        print(f"Optimized time:  {optimized_time:.1f}s")
        print(f"Original time:   {original_time:.1f}s")
        print(f"Speedup:         {speedup:.1f}x faster")
        print()
        
        if speedup >= 5:
            print("🚀 Excellent! Optimizations are working great!")
        elif speedup >= 2:
            print("✓ Good speedup achieved!")
        else:
            print("⚠ Speedup lower than expected. Check configuration.")
    elif optimized_time:
        print(f"Optimized time:  {optimized_time:.1f}s")
        print("Original method failed or skipped.")
    elif original_time:
        print(f"Original time:   {original_time:.1f}s")
        print("Optimized method failed.")
    else:
        print("Both methods failed.")
    
    print()
    print("=" * 80)
    
    # Cleanup instructions
    print()
    print("Cleanup:")
    print(f"  Delete test collections with: python -c \"from tools.rag import RAGTool; t = RAGTool('{username}'); t.delete_collection('{collection_name}_optimized'); t.delete_collection('{collection_name}_original')\"")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python test_rag_upload_performance.py <path_to_pdf>")
        print()
        print("Example:")
        print("  python test_rag_upload_performance.py test_document.pdf")
        print("  python test_rag_upload_performance.py /path/to/large_manual.pdf")
        sys.exit(1)
    
    pdf_path = sys.argv[1]
    test_upload_performance(pdf_path)
