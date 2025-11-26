# provenance.py

import os
import pymysql
from pymysql.cursors import DictCursor
import json
from typing import Optional, Dict, Any


class ProvenanceDB:
    def __init__(self):
        """
        Conecta ao MariaDB rodando no Docker.

        Como o MariaDB está rodando na porta 3307 no host,
        configuramos host=localhost e port=3307.
        """
        self.conn = pymysql.connect(
            host=os.getenv("PROV_DB_HOST", "localhost"),
            port=int(os.getenv("PROV_DB_PORT", "3307")),
            user=os.getenv("PROV_DB_USER", "larissa"),
            password=os.getenv("PROV_DB_PASSWORD", "1234"),
            database=os.getenv("PROV_DB_NAME", "provdb"),
            cursorclass=DictCursor,
            autocommit=True,
        )

    # ============================================================
    # EXPERIMENT
    # ============================================================

    def create_experiment(self, hotpot_path: str, seed: int, n_samples: int) -> int:
        query = """
            INSERT INTO experiment (hotpot_path, seed, n_samples)
            VALUES (%s, %s, %s)
        """
        with self.conn.cursor() as cur:
            cur.execute(query, (hotpot_path, seed, n_samples))
            return cur.lastrowid
        
    def update_experiment_hotpot_info(self, experiment_id: int,
                                      hotpot_path: str,
                                      seed: int,
                                      n_samples: int) -> None:
        query = """
            UPDATE experiment
            SET hotpot_path = %s,
                seed = %s,
                n_samples = %s
            WHERE id = %s
        """
        with self.conn.cursor() as cur:
            cur.execute(query, (hotpot_path, seed, n_samples, experiment_id))


    # ============================================================
    # CREATION (geração do dataset de explicabilidade)
    # ============================================================

    def create_creation(self,
                        experiment_id: int,
                        hotpotqa_sample: int,
                        prompt: str,
                        model: str,
                        temperature: float) -> int:
        """
        Cria um registro na tabela 'creation' e retorna o ID.
        """
        query = """
            INSERT INTO creation (experiment_id, hotpotqa_sample, prompt, model, temperature)
            VALUES (%s, %s, %s, %s, %s)
        """

        with self.conn.cursor() as cur:
            cur.execute(
                query,
                (experiment_id, hotpotqa_sample, prompt, model, temperature),
            )
            return cur.lastrowid

    # ============================================================
    # XAI DATASET ROW
    # ============================================================

    def insert_xai_row(self,
                       creation_id: int,
                       sample_id: str,
                       original_dataset_name: str,
                       question: str,
                       answer: str,
                       chunk: str,
                       explanation: str,
                       meta: Optional[Dict[str, Any]] = None) -> int:

        query = """
            INSERT INTO xai_dataset
            (creation_id, sample_id, original_dataset_name, question, answer, chunk, explanation, meta)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
        """

        meta_json = json.dumps(meta or {})

        with self.conn.cursor() as cur:
            cur.execute(
                query,
                (
                    creation_id,
                    str(sample_id),
                    original_dataset_name,
                    question,
                    answer,
                    chunk,
                    explanation,
                    meta_json,
                ),
            )
            return cur.lastrowid

    # ============================================================
    # VALIDATION
    # ============================================================

    def insert_validation(self, experiment_id: int, embedding_model: str, threshold: float,
                          is_valid: bool, details: Optional[Dict[str, Any]] = None) -> int:

        query = """
            INSERT INTO validation (experiment_id, embedding_model, threshold, is_valid, details)
            VALUES (%s, %s, %s, %s, %s)
        """

        details_json = json.dumps(details or {})

        with self.conn.cursor() as cur:
            cur.execute(query, (experiment_id, embedding_model, threshold, is_valid, details_json))
            return cur.lastrowid

    # ============================================================
    # COSINE RESULTS
    # ============================================================

    def insert_cosine_result(self,
                             experiment_id: int,
                             sample_id: str,
                             label: str,
                             cosine: float,
                             metadata: Optional[Dict[str, Any]] = None) -> int:
        """
        Registra a similaridade de Cosine para UMA explicação
        (uma linha do dataset flatten).
        """
        query = """
            INSERT INTO cosine_results
            (experiment_id, sample_id, label, cosine, metadata)
            VALUES (%s, %s, %s, %s, %s)
        """

        metadata_json = json.dumps(metadata or {})

        with self.conn.cursor() as cur:
            cur.execute(
                query,
                (experiment_id, str(sample_id), str(label), float(cosine), metadata_json),
            )
            return cur.lastrowid

    # ============================================================
    # JACCARD RESULTS
    # ============================================================

    def insert_jaccard_result(self,
                              experiment_id: int,
                              sample_id: str,
                              label: str,
                              jaccard: float,
                              metadata: Optional[Dict[str, Any]] = None) -> int:
        """
        Registra a similaridade de Jaccard para UMA explicação
        (uma linha do dataset flatten).
        """
        query = """
            INSERT INTO jaccard_results
            (experiment_id, sample_id, label, jaccard, metadata)
            VALUES (%s, %s, %s, %s, %s)
        """

        metadata_json = json.dumps(metadata or {})

        with self.conn.cursor() as cur:
            cur.execute(
                query,
                (experiment_id, str(sample_id), str(label), float(jaccard), metadata_json),
            )
            return cur.lastrowid
    # ============================================================
    # LOGIC METRIC CONFIG
    # ============================================================

    def insert_logic_metric(self, experiment_id: int,
                            num_trials: int,
                            predicate_config: Dict,
                            rules_config: Dict) -> int:

        query = """
            INSERT INTO logic_metric
            (experiment_id, num_trials, predicate_config, rules_config)
            VALUES (%s, %s, %s, %s)
        """

        with self.conn.cursor() as cur:
            cur.execute(query, (
                experiment_id,
                num_trials,
                json.dumps(predicate_config),
                json.dumps(rules_config),
            ))
            return cur.lastrowid

        # ============================================================
    # LOGIC METRIC (configuração da métrica lógica)
    # ============================================================

    def create_logic_metric(self,
                            experiment_id: int,
                            num_trials: int,
                            predicate_config: Optional[Dict[str, Any]] = None,
                            rules_config: Optional[Dict[str, Any]] = None,
                            facts_config: Optional[Dict[str, Any]] = None) -> int:
        """
        Cria um registro na tabela logic_metric para um experimento.
        Você pode criar já com alguns configs ou com tudo None e ir atualizando depois.
        """
        query = """
            INSERT INTO logic_metric (experiment_id, num_trials, predicate_config, rules_config, facts_config)
            VALUES (%s, %s, %s, %s, %s)
        """

        with self.conn.cursor() as cur:
            cur.execute(
                query,
                (
                    experiment_id,
                    num_trials,
                    json.dumps(predicate_config or {}),
                    json.dumps(rules_config or {}),
                    json.dumps(facts_config or {}),
                ),
            )
            return cur.lastrowid

    def update_logic_metric_configs(self,
                                    logic_metric_id: int,
                                    predicate_config: Optional[Dict[str, Any]] = None,
                                    rules_config: Optional[Dict[str, Any]] = None,
                                    facts_config: Optional[Dict[str, Any]] = None) -> None:
        """
        Atualiza parcialmente os configs de logic_metric.
        Você pode chamar várias vezes (ex: depois do script de predicados, depois do de regras, etc.).
        Só os parâmetros não-nulos serão atualizados.
        """
        sets = []
        params = []

        if predicate_config is not None:
            sets.append("predicate_config = %s")
            params.append(json.dumps(predicate_config))

        if rules_config is not None:
            sets.append("rules_config = %s")
            params.append(json.dumps(rules_config))

        if facts_config is not None:
            sets.append("facts_config = %s")
            params.append(json.dumps(facts_config))

        if not sets:
            return  # nada pra atualizar

        query = f"""
            UPDATE logic_metric
            SET {", ".join(sets)}
            WHERE id = %s
        """
        params.append(logic_metric_id)

        with self.conn.cursor() as cur:
            cur.execute(query, tuple(params))

    
        # ============================================================
    # LOGIC RESULT (por explicação e trial)
    # ============================================================

    def insert_logic_result(self,
                            experiment_id: int,
                            logic_metric_id: int,
                            trial_number: int,
                            sample_id: str,
                            label: str,
                            precision_result: float,
                            recall_result: float,
                            f1_result: float,
                            facts: Optional[Dict[str, Any]] = None,
                            metadata: Optional[Dict[str, Any]] = None) -> int:
        """
        Registra o resultado da métrica lógica para UMA explicação em UM trial.
        """
        query = """
            INSERT INTO logic_result
            (experiment_id, logic_metric_id, trial_number,
             sample_id, label,
             precision_result, recall_result, f1_result,
             facts, metadata)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """

        facts_json = json.dumps(facts or {})
        metadata_json = json.dumps(metadata or {})

        with self.conn.cursor() as cur:
            cur.execute(
                query,
                (
                    experiment_id,
                    logic_metric_id,
                    trial_number,
                    str(sample_id),
                    str(label),
                    float(precision_result),
                    float(recall_result),
                    float(f1_result),
                    facts_json,
                    metadata_json,
                ),
            )
            return cur.lastrowid


    # ============================================================
    # CLOSE CONNECTION
    # ============================================================

    def close(self):
        self.conn.close()
