CREATE DATABASE IF NOT EXISTS business_code_agent
  DEFAULT CHARACTER SET utf8mb4
  DEFAULT COLLATE utf8mb4_unicode_ci;

USE business_code_agent;

CREATE TABLE IF NOT EXISTS project (
  id            VARCHAR(64)  NOT NULL PRIMARY KEY,
  name          VARCHAR(128) NOT NULL,
  root_path     VARCHAR(512) NOT NULL,
  description   TEXT NULL,
  created_at    DATETIME     NOT NULL DEFAULT CURRENT_TIMESTAMP,
  updated_at    DATETIME     NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

CREATE TABLE IF NOT EXISTS project_meta (
  project_id    VARCHAR(64)  NOT NULL PRIMARY KEY,
  tech_stack    TEXT NULL,
  overview_json JSON NULL,
  created_at    DATETIME     NOT NULL DEFAULT CURRENT_TIMESTAMP,
  updated_at    DATETIME     NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

CREATE TABLE IF NOT EXISTS ingest_job (
  id            VARCHAR(64)  NOT NULL PRIMARY KEY,
  project_id    VARCHAR(64)  NOT NULL,
  status        VARCHAR(32)  NOT NULL,
  stage         VARCHAR(64)  NULL,
  total_files   INT          NOT NULL DEFAULT 0,
  processed     INT          NOT NULL DEFAULT 0,
  message       TEXT         NULL,
  error         TEXT         NULL,
  created_at    DATETIME     NOT NULL DEFAULT CURRENT_TIMESTAMP,
  updated_at    DATETIME     NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
  INDEX idx_ingest_project (project_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

CREATE TABLE IF NOT EXISTS symbol (
  id            VARCHAR(64)  NOT NULL PRIMARY KEY,
  project_id    VARCHAR(64)  NOT NULL,
  module_name   VARCHAR(128) NULL,
  kind          VARCHAR(64)  NOT NULL,
  fqcn          VARCHAR(512) NOT NULL,
  simple_name   VARCHAR(256) NOT NULL,
  method_name   VARCHAR(256) NULL,
  method_sig    VARCHAR(1024) NULL,
  file_path     VARCHAR(1024) NOT NULL,
  start_line    INT          NULL,
  end_line      INT          NULL,
  annotations   JSON         NULL,
  modifiers     JSON         NULL,
  parent_fqcn   VARCHAR(512) NULL,
  raw_code      MEDIUMTEXT   NULL,
  created_at    DATETIME     NOT NULL DEFAULT CURRENT_TIMESTAMP,
  updated_at    DATETIME     NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
  UNIQUE KEY uk_symbol (project_id, fqcn(180), method_sig(180)),
  INDEX idx_symbol_project (project_id),
  INDEX idx_symbol_kind (kind),
  INDEX idx_symbol_fqcn (fqcn(180)),
  INDEX idx_symbol_module (module_name)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

CREATE TABLE IF NOT EXISTS tag_dict (
  id            BIGINT       NOT NULL AUTO_INCREMENT PRIMARY KEY,
  project_id    VARCHAR(64)  NOT NULL,
  tag           VARCHAR(128) NOT NULL,
  category      VARCHAR(64)  NULL,
  description   VARCHAR(512) NULL,
  created_at    DATETIME     NOT NULL DEFAULT CURRENT_TIMESTAMP,
  UNIQUE KEY uk_tag (project_id, tag),
  INDEX idx_tag_project (project_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

CREATE TABLE IF NOT EXISTS symbol_tag (
  id            BIGINT       NOT NULL AUTO_INCREMENT PRIMARY KEY,
  project_id    VARCHAR(64)  NOT NULL,
  symbol_id     VARCHAR(64)  NOT NULL,
  tag_id        BIGINT       NOT NULL,
  created_at    DATETIME     NOT NULL DEFAULT CURRENT_TIMESTAMP,
  UNIQUE KEY uk_symbol_tag (symbol_id, tag_id),
  INDEX idx_st_project (project_id),
  INDEX idx_st_tag (tag_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

CREATE TABLE IF NOT EXISTS symbol_desc (
  id            BIGINT       NOT NULL AUTO_INCREMENT PRIMARY KEY,
  project_id    VARCHAR(64)  NOT NULL,
  symbol_id     VARCHAR(64)  NOT NULL,
  summary       MEDIUMTEXT   NOT NULL,
  summary_business_clear  MEDIUMTEXT NULL,
  summary_business_detail MEDIUMTEXT NULL,
  summary_dev_clear       MEDIUMTEXT NULL,
  summary_dev_detail      MEDIUMTEXT NULL,
  business_domain VARCHAR(128) NULL,
  extra_json    JSON         NULL,
  created_at    DATETIME     NOT NULL DEFAULT CURRENT_TIMESTAMP,
  updated_at    DATETIME     NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
  UNIQUE KEY uk_symbol_desc (symbol_id),
  INDEX idx_sd_project (project_id),
  INDEX idx_sd_domain (business_domain)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

CREATE TABLE IF NOT EXISTS chat_session (
  id            VARCHAR(64)  NOT NULL PRIMARY KEY,
  project_id    VARCHAR(64)  NOT NULL,
  title         VARCHAR(256) NULL,
  context_json  JSON         NULL,
  created_at    DATETIME     NOT NULL DEFAULT CURRENT_TIMESTAMP,
  updated_at    DATETIME     NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
  INDEX idx_cs_project (project_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

CREATE TABLE IF NOT EXISTS chat_message (
  id            BIGINT       NOT NULL AUTO_INCREMENT PRIMARY KEY,
  session_id    VARCHAR(64)  NOT NULL,
  role          VARCHAR(32)  NOT NULL,
  content       MEDIUMTEXT   NOT NULL,
  meta_json     JSON         NULL,
  created_at    DATETIME     NOT NULL DEFAULT CURRENT_TIMESTAMP,
  INDEX idx_cm_session (session_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

CREATE TABLE IF NOT EXISTS chat_feedback (
  id                    BIGINT       NOT NULL AUTO_INCREMENT PRIMARY KEY,
  project_id            VARCHAR(64)  NOT NULL,
  session_id            VARCHAR(64)  NOT NULL,
  assistant_message_id  BIGINT       NOT NULL,
  vote                  VARCHAR(16)  NOT NULL,
  comment               TEXT         NULL,
  question_excerpt      TEXT         NULL,
  reply_excerpt         TEXT         NULL,
  conversation_json     JSON         NULL,
  created_at            DATETIME     NOT NULL DEFAULT CURRENT_TIMESTAMP,
  updated_at            DATETIME     NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
  UNIQUE KEY uk_feedback_msg (session_id, assistant_message_id),
  INDEX idx_cf_project (project_id),
  INDEX idx_cf_session (session_id),
  INDEX idx_cf_created (created_at)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
