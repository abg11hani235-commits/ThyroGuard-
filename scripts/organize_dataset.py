import os
import shutil
import hashlib
import json
import pypdf

base_dir = r'd:/RAG/ThyroGuard_Dataset'
raw_dir = os.path.join(base_dir, 'raw')

# Define target structure and mapping
file_mapping = {
    'ata_2014_hypothyroidism_guidelines.pdf': {
        'target_subpath': 'Guidelines/Hypothyroidism/ATA_2014_Hypothyroidism_Treatment_Guidelines.pdf',
        'doc_slug': 'ata_2014_hypo',
        'title': 'Guidelines for the Treatment of Hypothyroidism: Prepared by the American Thyroid Association Task Force on Thyroid Hormone Replacement',
        'authoring_body': 'American Thyroid Association (ATA)',
        'year': 2014,
        'category': 'Clinical Practice Guideline',
        'subtopic': 'Hypothyroidism',
        'evidence_level': 'Grade A/B Evidence-Based Guideline'
    },
    '308682175.pdf': {
        'target_subpath': 'Guidelines/Hypothyroidism/ATA_AACE_2012_Hypothyroidism_Adults_Guidelines.pdf',
        'doc_slug': 'ata_aace_2012_hypo',
        'title': 'Clinical Practice Guidelines for Hypothyroidism in Adults: Cosponsored by the American Association of Clinical Endocrinologists and the American Thyroid Association',
        'authoring_body': 'AACE & ATA',
        'year': 2012,
        'category': 'Clinical Practice Guideline',
        'subtopic': 'Hypothyroidism',
        'evidence_level': 'Clinical Practice Guidelines'
    },
    'hyperthyroidism_guidelines_2016 (1).pdf': {
        'target_subpath': 'Guidelines/Hyperthyroidism/ATA_2016_Hyperthyroidism_and_Thyrotoxicosis_Guidelines.pdf',
        'doc_slug': 'ata_2016_hyper',
        'title': '2016 American Thyroid Association Guidelines for Diagnosis and Management of Hyperthyroidism and Other Causes of Thyrotoxicosis',
        'authoring_body': 'American Thyroid Association (ATA)',
        'year': 2016,
        'category': 'Clinical Practice Guideline',
        'subtopic': 'Hyperthyroidism & Thyrotoxicosis',
        'evidence_level': 'Grade A/B Evidence-Based Guideline'
    },
    'a621984c3d1c852d8de9561b84ec27d3.pdf': {
        'target_subpath': 'Guidelines/Thyroid_Nodules/ATA_2015_Thyroid_Nodules_and_DTC_Guidelines.pdf',
        'doc_slug': 'ata_2015_nodules_dtc',
        'title': '2015 American Thyroid Association Management Guidelines for Adult Patients with Thyroid Nodules and Differentiated Thyroid Cancer',
        'authoring_body': 'American Thyroid Association (ATA)',
        'year': 2015,
        'category': 'Clinical Practice Guideline',
        'subtopic': 'Thyroid Nodules & Differentiated Thyroid Cancer',
        'evidence_level': 'Comprehensive Clinical Guideline'
    },
    'ATC-Diagnosis.pdf': {
        'target_subpath': 'Guidelines/Thyroid_Cancer/ATA_2020_Anaplastic_Thyroid_Cancer_Diagnosis.pdf',
        'doc_slug': 'ata_2020_atc',
        'title': '2020 ATA Guidelines for Management of Patients with Anaplastic Thyroid Cancer: Diagnosis, Cytology, and Histopathology',
        'authoring_body': 'American Thyroid Association (ATA)',
        'year': 2020,
        'category': 'Clinical Practice Guideline',
        'subtopic': 'Anaplastic Thyroid Cancer (ATC)',
        'evidence_level': 'Clinical Guideline / Diagnostic Protocol'
    },
    '102661_CC-BY.pdf': {
        'target_subpath': 'Reviews/Review_2021_Thyroid_Cancer_Markers_and_Treatment.pdf',
        'doc_slug': 'review_2021_thyroid_cancer',
        'title': 'Thyroid Cancer: Epidemiology, Classification, Risk Factors, Diagnostic and Prognostic Markers, and Current Treatment Strategies',
        'authoring_body': 'Medical University of Lublin / Cancers Journal',
        'year': 2021,
        'category': 'Systematic Review / Journal Article',
        'subtopic': 'Thyroid Cancer Comprehensive Review',
        'evidence_level': 'Peer-Reviewed Scientific Review'
    },
    'Hypothyroidism_web_booklet.pdf': {
        'target_subpath': 'References/Hypothyroidism_Patient_Information_Booklet.pdf',
        'doc_slug': 'patient_hypo_booklet',
        'title': 'Hypothyroidism Patient Information and Management Guide',
        'authoring_body': 'Thyroid Foundation / Patient Education Series',
        'year': 2018,
        'category': 'Patient Education & Reference',
        'subtopic': 'Patient Information',
        'evidence_level': 'Patient Educational Reference'
    },
    'ThyroidDisease.pdf': {
        'target_subpath': 'References/Thyroid_Disease_Patient_Guide_Ribbink.pdf',
        'doc_slug': 'patient_thyroid_guide_ribbink',
        'title': 'Thyroid Disease: A Patients Clinical Guide',
        'authoring_body': 'J. Van Lier Ribbink, M.D., F.A.C.S.',
        'year': 2012,
        'category': 'Clinical Reference Guide',
        'subtopic': 'General Thyroid Disorders',
        'evidence_level': 'Clinical Reference'
    }
}

inventory = []

for src_name, info in file_mapping.items():
    src_file = os.path.join(raw_dir, src_name)
    if not os.path.exists(src_file):
        print(f"Warning: {src_name} not found in raw_dir")
        continue
    
    dest_file = os.path.join(base_dir, info['target_subpath'])
    os.makedirs(os.path.dirname(dest_file), exist_ok=True)
    shutil.copy2(src_file, dest_file)
    
    with open(dest_file, 'rb') as f:
        content = f.read()
        sha256 = hashlib.sha256(content).hexdigest()
        file_size_kb = round(len(content) / 1024, 1)
    
    reader = pypdf.PdfReader(dest_file)
    page_count = len(reader.pages)
    
    item = {
        'doc_slug': info['doc_slug'],
        'filename': os.path.basename(dest_file),
        'relative_path': info['target_subpath'],
        'title': info['title'],
        'authoring_body': info['authoring_body'],
        'year': info['year'],
        'category': info['category'],
        'subtopic': info['subtopic'],
        'evidence_level': info['evidence_level'],
        'page_count': page_count,
        'file_size_kb': file_size_kb,
        'sha256': sha256
    }
    inventory.append(item)
    print(f"Organized: {info['doc_slug']} ({page_count} pages) -> {info['target_subpath']}")

os.makedirs('d:/RAG/data/processed', exist_ok=True)
json_path = 'd:/RAG/data/processed/document_inventory.json'
with open(json_path, 'w', encoding='utf-8') as f:
    json.dump(inventory, f, indent=2, ensure_ascii=False)

# Generate Markdown Document Inventory
md_path = 'd:/RAG/DOCUMENT_INVENTORY.md'
with open(md_path, 'w', encoding='utf-8') as f:
    f.write("# ThyroGuard Medical RAG — Document Inventory\n\n")
    f.write("This document provides the authoritative catalog of all clinical guidelines, systematic reviews, and reference documents ingested into the **ThyroGuard Knowledge Base**.\n\n")
    f.write("## 1. Corpus Summary\n\n")
    f.write(f"- **Total Documents Ingested**: {len(inventory)}\n")
    f.write(f"- **Total Pages**: {sum(d['page_count'] for d in inventory)}\n")
    f.write(f"- **Total Size**: {round(sum(d['file_size_kb'] for d in inventory) / 1024, 2)} MB\n")
    f.write(f"- **Clinical Scope**: Hypothyroidism, Hyperthyroidism & Thyrotoxicosis, Thyroid Nodules, Differentiated Thyroid Cancer (DTC), and Anaplastic Thyroid Cancer (ATC).\n\n")
    f.write("## 2. Ingested Documents Registry\n\n")
    f.write("| Slug (`doc_slug`) | Document Title | Authority & Year | Clinical Subtopic | Pages | File Size (KB) |\n")
    f.write("|---|---|---|---|---:|---:|\n")
    for d in inventory:
        f.write(f"| `{d['doc_slug']}` | **{d['title']}** | {d['authoring_body']} ({d['year']}) | {d['subtopic']} | {d['page_count']} | {d['file_size_kb']} |\n")
    f.write("\n## 3. Cryptographic Verification & Audit Hashes\n\n")
    f.write("| Slug | Target Relative Path | SHA256 Hash |\n")
    f.write("|---|---|---|\n")
    for d in inventory:
        f.write(f"| `{d['doc_slug']}` | `{d['relative_path']}` | `{d['sha256']}` |\n")

print(f"\nDocument inventory generated with {len(inventory)} documents.")
