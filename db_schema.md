# Database Schema: russian_ukrainian_war
# Generated on: 2025-05-09 03:05:21


## Table of Contents
- [alembic_version](#alembic_version)
- [classifications](#classifications)
- [document_section](#document_section)
- [document_section_chunk](#document_section_chunk)
- [logs](#logs)
- [taxonomy](#taxonomy)
- [taxonomy_clusters_kmeans](#taxonomy_clusters_kmeans)
- [taxonomy_granular](#taxonomy_granular)
- [uploaded_document](#uploaded_document)

---

## Database Summary
- Tables: 9
- Total Columns: 74
- Total Foreign Keys: 7
- Total Indexes: 13
- Total Rows: 2,922,447

---

## alembic_version
<a id="alembic_version"></a>

**Row count**: 1

### Columns
| Column | Type | Nullable | Default |
|--------|------|----------|---------|
| **version_num** (PK) | character varying | NO | None |

### Primary Key
- **Columns**: version_num

### Indexes
- **alembic_version_pkc**: UNIQUE (version_num)

---

## classifications
<a id="classifications"></a>

**Row count**: 161,296

### Columns
| Column | Type | Nullable | Default |
|--------|------|----------|---------|
| **id** (PK) | integer | NO | nextval('classifications_id_seq'::regclass) |
| chunk_id (FK) | integer | NO | None |
| chat_id | character varying | NO | None |
| model | character varying | NO | None |
| created_at | timestamp without time zone | NO | None |
| done_reason | character varying | NO | None |
| total_duration | bigint | NO | None |
| usage_prompt_tokens | integer | NO | None |
| usage_completion_tokens | integer | NO | None |
| usage_tokens | integer | NO | None |
| request | text | NO | None |
| raw_response | text | YES | None |
| response | text | YES | None |

### Primary Key
- **Columns**: id

### Foreign Keys
- **classifications_chunk_id_fkey**
  - **Columns**: chunk_id
  - **References**: document_section_chunk(id)

---

## document_section
<a id="document_section"></a>

**Row count**: 524,964

### Columns
| Column | Type | Nullable | Default |
|--------|------|----------|---------|
| **id** (PK) | integer | NO | nextval('document_section_id_seq'::regclass) |
| uploaded_document_id (FK) | integer | NO | None |
| parent_document_section_id (FK) | integer | YES | None |
| heading_title | text | NO | None |
| heading_depth | integer | NO | None |
| content | text | YES | None |
| sequence_number | integer | NO | None |
| path | USER-DEFINED | NO | None |

### Primary Key
- **Columns**: id

### Foreign Keys
- **document_section_parent_document_section_id_fkey**
  - **Columns**: parent_document_section_id
  - **References**: document_section(id)
- **document_section_uploaded_document_id_fkey**
  - **Columns**: uploaded_document_id
  - **References**: uploaded_document(id)

### Indexes
- **idx_document_section_path**: (path)
- **idx_document_section_uploaded_document_id**: (uploaded_document_id)

---

## document_section_chunk
<a id="document_section_chunk"></a>

**Row count**: 473,734

### Columns
| Column | Type | Nullable | Default |
|--------|------|----------|---------|
| **id** (PK) | integer | NO | nextval('document_section_chunk_id_seq'::regclass) |
| document_section_id (FK) | integer | NO | None |
| chunk_index | integer | NO | None |
| content | text | NO | None |
| embedding | USER-DEFINED | YES | None |
| embedding_model | text | YES | None |
| named_entities | jsonb | YES | None |
| keywords | ARRAY | YES | None |

### Primary Key
- **Columns**: id

### Foreign Keys
- **document_section_chunk_document_section_id_fkey**
  - **Columns**: document_section_id
  - **References**: document_section(id)

### Indexes
- **idx_embedding**: (embedding)
- **idx_document_section_chunk_document_section_id**: (document_section_id)

---

## logs
<a id="logs"></a>

**Row count**: 548,725

### Columns
| Column | Type | Nullable | Default |
|--------|------|----------|---------|
| **id** (PK) | integer | NO | nextval('logs_id_seq'::regclass) |
| document_id | text | YES | None |
| database | character varying | YES | None |
| stage | character varying | YES | None |
| model | character varying | YES | None |
| created_at | timestamp without time zone | YES | None |
| done | boolean | YES | None |
| done_reason | character varying | YES | None |
| total_duration | bigint | YES | None |
| load_duration | bigint | YES | None |
| prompt_eval_count | integer | YES | None |
| prompt_eval_duration | bigint | YES | None |
| eval_count | integer | YES | None |
| eval_duration | bigint | YES | None |
| raw_response | text | YES | None |
| response | text | YES | None |

### Primary Key
- **Columns**: id

---

## taxonomy
<a id="taxonomy"></a>

**Row count**: 213,965

### Columns
| Column | Type | Nullable | Default |
|--------|------|----------|---------|
| **id** (PK) | integer | NO | nextval('taxonomy_id_seq'::regclass) |
| classification_id (FK) | integer | NO | None |
| chunk_id (FK) | integer | NO | None |
| category | character varying | NO | None |
| subcategory | character varying | NO | None |
| sub_subcategory | character varying | NO | None |
| taxonomy_reasoning | text | NO | None |
| chunk_level_reasoning | text | NO | None |

### Primary Key
- **Columns**: id

### Foreign Keys
- **taxonomy_chunk_id_fkey**
  - **Columns**: chunk_id
  - **References**: document_section_chunk(id)
- **taxonomy_classification_id_fkey**
  - **Columns**: classification_id
  - **References**: classifications(id)

### Indexes
- **idx_taxonomy_chunk_id**: (chunk_id)
- **idx_taxonomy_category**: (category)
- **idx_taxonomy_subcategory**: (subcategory)
- **idx_taxonomy_sub_subcategory**: (sub_subcategory)

---

## taxonomy_clusters_kmeans
<a id="taxonomy_clusters_kmeans"></a>

**Row count**: 309

### Columns
| Column | Type | Nullable | Default |
|--------|------|----------|---------|
| **id** (PK) | integer | NO | nextval('taxonomy_clusters_kmeans_id_seq'::regclass) |
| taxonomic_element | character varying | NO | None |
| cluster_id | character varying | NO | None |
| cluster_label | character varying | YES | None |
| span_ids | ARRAY | NO | None |
| summary | json | YES | None |

### Primary Key
- **Columns**: id

### Indexes
- **uq_tax_element_cluster_kmeans**: UNIQUE (taxonomic_element, cluster_id)
- **ix_taxonomy_clusters_kmeans_id**: (id)

---

## taxonomy_granular
<a id="taxonomy_granular"></a>

**Row count**: 686,259

### Columns
| Column | Type | Nullable | Default |
|--------|------|----------|---------|
| **id** (PK) | integer | NO | nextval('taxonomy_granular_id_seq'::regclass) |
| chunk_te_id (FK) | integer | NO | None |
| text_span | text | NO | None |
| justification | text | NO | None |
| embedding | USER-DEFINED | YES | None |

### Primary Key
- **Columns**: id

### Foreign Keys
- **taxonomy_granular_chunk_te_id_fkey**
  - **Columns**: chunk_te_id
  - **References**: taxonomy(id)

---

## uploaded_document
<a id="uploaded_document"></a>

**Row count**: 313,194

### Columns
| Column | Type | Nullable | Default |
|--------|------|----------|---------|
| **id** (PK) | integer | NO | nextval('uploaded_document_id_seq'::regclass) |
| document_id | character varying | NO | None |
| database | character varying | NO | None |
| date | date | YES | None |
| author | character varying | YES | None |
| language | character varying | YES | None |
| is_full_text_present | boolean | NO | None |
| text | text | YES | None |
| text_en | text | YES | None |

### Primary Key
- **Columns**: id

### Indexes
- **idx_uploaded_document_language**: (language)
- **idx_uploaded_document_database**: (database)

---
