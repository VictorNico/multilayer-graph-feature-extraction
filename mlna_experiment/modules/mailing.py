#!/usr/bin/env python3
"""
Script pour compiler un fichier LaTeX et l'envoyer par email via Gmail SMTP
"""

import os
import sys
import subprocess
import smtplib
import logging
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders
from pathlib import Path

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class LaTeXEmailSender:
    def __init__(self, gmail_user, gmail_password, display_name="Pipeline MLNA"):
        """
        Initialise le gestionnaire LaTeX/Email

        Args:
            gmail_user (str): Adresse Gmail
            gmail_password (str): Mot de passe d'application Gmail
            display_name (str): Sender name
        """
        self.gmail_user = gmail_user
        self.gmail_password = gmail_password
        self.display_name = display_name

    def compile_latex(self, tex_file_path, output_dir=None, clean_temp=True):
        """
        Compile un fichier LaTeX en PDF

        Args:
            tex_file_path (str): Chemin vers le fichier .tex
            output_dir (str): Répertoire de sortie (optionnel)
            clean_temp (bool): Nettoyer les fichiers temporaires

        Returns:
            str: Chemin vers le fichier PDF généré
        """
        tex_path = Path(tex_file_path)

        if not tex_path.exists():
            raise FileNotFoundError(f"Fichier LaTeX non trouvé: {tex_file_path}")

        # Répertoire de travail
        work_dir = tex_path.parent if output_dir is None else Path(output_dir)
        work_dir.mkdir(parents=True, exist_ok=True)

        # Nom du fichier sans extension
        base_name = tex_path.stem
        pdf_path = work_dir / f"{base_name}.pdf"

        logger.info(f"Compilation de {tex_file_path}")

        try:
            # Première compilation avec gestion d'encodage
            logger.info("Première compilation LaTeX...")
            result = subprocess.run([
                'pdflatex',
                '-interaction=nonstopmode',
                f'-output-directory={work_dir}',
                str(tex_path)
            ], capture_output=True, text=False, cwd=work_dir)  # text=False pour éviter l'erreur d'encodage

            # Décoder la sortie manuellement avec gestion d'erreur
            try:
                stdout_decoded = result.stdout.decode('utf-8')
                stderr_decoded = result.stderr.decode('utf-8')
            except UnicodeDecodeError:
                # Fallback avec remplacement des caractères problématiques
                stdout_decoded = result.stdout.decode('utf-8', errors='replace')
                stderr_decoded = result.stderr.decode('utf-8', errors='replace')
                logger.warning("Caractères d'encodage remplacés dans la sortie pdflatex")

            if result.returncode != 0:
                logger.error("Erreur lors de la première compilation:")
                logger.error("STDOUT:")
                logger.error(stdout_decoded)
                logger.error("STDERR:")
                logger.error(stderr_decoded)

                # Essayer de lire le fichier .log pour plus de détails
                log_file = work_dir / f"{base_name}.log"
                if log_file.exists():
                    try:
                        with open(log_file, 'r', encoding='utf-8', errors='replace') as f:
                            log_content = f.read()
                            logger.error("Contenu du fichier .log (dernières 1000 caractères):")
                            logger.error(log_content[-1000:])
                    except Exception as log_e:
                        logger.warning(f"Impossible de lire le fichier .log: {log_e}")

                raise RuntimeError("Échec de la compilation LaTeX")

            logger.info("Première compilation réussie")

            # Deuxième compilation (pour les références croisées)
            logger.info("Deuxième compilation pour les références...")
            result = subprocess.run([
                'pdflatex',
                '-interaction=nonstopmode',
                f'-output-directory={work_dir}',
                str(tex_path)
            ], capture_output=True, text=False, cwd=work_dir)  # text=False

            # Décoder la sortie de la deuxième compilation
            try:
                stdout_decoded = result.stdout.decode('utf-8')
                stderr_decoded = result.stderr.decode('utf-8')
            except UnicodeDecodeError:
                stdout_decoded = result.stdout.decode('utf-8', errors='replace')
                stderr_decoded = result.stderr.decode('utf-8', errors='replace')

            if result.returncode != 0:
                logger.warning("Avertissement lors de la deuxième compilation:")
                logger.warning("STDOUT:")
                logger.warning(stdout_decoded)
                logger.warning("STDERR:")
                logger.warning(stderr_decoded)
            else:
                logger.info("Deuxième compilation réussie")

            # Nettoyage des fichiers temporaires
            if clean_temp:
                self._clean_temp_files(work_dir, base_name)

            if not pdf_path.exists():
                raise FileNotFoundError(f"Le fichier PDF n'a pas été généré: {pdf_path}")

            logger.info(f"PDF généré avec succès: {pdf_path}")
            return str(pdf_path)

        except subprocess.CalledProcessError as e:
            logger.error(f"Erreur lors de l'exécution de pdflatex: {e}")
            raise
        except UnicodeDecodeError as e:
            logger.error(f"Erreur d'encodage lors de la lecture de la sortie pdflatex: {e}")
            logger.error("Le fichier LaTeX ou la sortie de pdflatex contient des caractères non-UTF-8")
            raise
        except Exception as e:
            logger.error(f"Erreur inattendue: {e}")
            raise

    # Version alternative encore plus robuste
    def compile_latex_robust(self, tex_file_path, output_dir=None, clean_temp=True):
        """
        Version encore plus robuste de la compilation LaTeX
        """
        tex_path = Path(tex_file_path)

        if not tex_path.exists():
            raise FileNotFoundError(f"Fichier LaTeX non trouvé: {tex_file_path}")

        # Répertoire de travail
        work_dir = tex_path.parent if output_dir is None else Path(output_dir)
        work_dir.mkdir(parents=True, exist_ok=True)

        # Nom du fichier sans extension
        base_name = tex_path.stem
        pdf_path = work_dir / f"{base_name}.pdf"

        logger.info(f"Compilation de {tex_file_path}")

        def run_pdflatex_safe(attempt_num):
            """Exécute pdflatex de manière sécurisée"""
            logger.info(f"Tentative de compilation #{attempt_num}")

            # Différentes stratégies d'encodage
            encodings_to_try = ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']

            for encoding in encodings_to_try:
                try:
                    result = subprocess.run([
                        'pdflatex',
                        '-interaction=nonstopmode',
                        f'-output-directory={work_dir}',
                        str(tex_path)
                    ], capture_output=True, cwd=work_dir,
                        encoding=encoding, errors='replace')

                    logger.info(f"Compilation réussie avec encodage: {encoding}")
                    return result

                except UnicodeDecodeError:
                    logger.warning(f"Encodage {encoding} échoué, essai suivant...")
                    continue
                except Exception as e:
                    logger.error(f"Erreur avec encodage {encoding}: {e}")
                    continue

            # Si tous les encodages échouent, utiliser bytes
            logger.warning("Tous les encodages ont échoué, utilisation du mode bytes")
            result = subprocess.run([
                'pdflatex',
                '-interaction=nonstopmode',
                f'-output-directory={work_dir}',
                str(tex_path)
            ], capture_output=True, cwd=work_dir)

            # Convertir manuellement
            result.stdout = result.stdout.decode('utf-8', errors='replace')
            result.stderr = result.stderr.decode('utf-8', errors='replace')

            return result

        try:
            # Première compilation
            result = run_pdflatex_safe(1)

            if result.returncode != 0:
                logger.error("Erreur lors de la première compilation:")
                logger.error("STDOUT:")
                logger.error(result.stdout)
                logger.error("STDERR:")
                logger.error(result.stderr)
                raise RuntimeError("Échec de la compilation LaTeX")

            logger.info("Première compilation réussie")

            # Deuxième compilation
            logger.info("Deuxième compilation pour les références...")
            result = run_pdflatex_safe(2)

            if result.returncode != 0:
                logger.warning("Avertissement lors de la deuxième compilation")
                logger.warning(result.stdout)
            else:
                logger.info("Deuxième compilation réussie")

            # Nettoyage des fichiers temporaires
            if clean_temp:
                self._clean_temp_files(work_dir, base_name)

            if not pdf_path.exists():
                raise FileNotFoundError(f"Le fichier PDF n'a pas été généré: {pdf_path}")

            logger.info(f"PDF généré avec succès: {pdf_path}")
            return str(pdf_path)

        except Exception as e:
            logger.error(f"Erreur lors de la compilation: {e}")
            raise

    def _clean_temp_files(self, work_dir, base_name):
        """Nettoie les fichiers temporaires LaTeX"""
        temp_extensions = ['.aux', '.log', '.out', '.toc', '.nav', '.snm', '.fls', '.fdb_latexmk']

        for ext in temp_extensions:
            temp_file = work_dir / f"{base_name}{ext}"
            if temp_file.exists():
                temp_file.unlink()
                logger.debug(f"Supprimé: {temp_file}")

    def send_email(self, to_emails, subject, body, pdf_path=None, tex_path=None, cc_emails=None):
        """
        Envoie un email avec le PDF et/ou le fichier TEX en pièces jointes

        Args:
            to_emails (list): Liste des destinataires
            subject (str): Sujet de l'email
            body (str): Corps de l'email
            pdf_path (str): Chemin vers le PDF à joindre
            tex_path (str): Chemin vers le fichier TEX à joindre
            cc_emails (list): Liste des destinataires en copie
        """
        msg = MIMEMultipart()
        if self.display_name:
            msg['From'] = f"{self.display_name} <{self.gmail_user}>"
        else:
            msg['From'] = self.gmail_user
        msg['To'] = ', '.join(to_emails)
        if cc_emails:
            msg['Cc'] = ', '.join(cc_emails)
        msg['Subject'] = subject

        # Corps de l'email
        msg.attach(MIMEText(body, 'plain', 'utf-8'))

        # Pièce jointe PDF
        if pdf_path and os.path.exists(pdf_path):
            with open(pdf_path, "rb") as attachment:
                part = MIMEBase('application', 'octet-stream')
                part.set_payload(attachment.read())

            encoders.encode_base64(part)
            filename = os.path.basename(pdf_path)
            part.add_header(
                'Content-Disposition',
                f'attachment; filename={filename}'
            )
            msg.attach(part)
            logger.info(f"PDF joint: {filename}")

        # Pièce jointe TEX
        if tex_path and os.path.exists(tex_path):
            with open(tex_path, "rb") as attachment:
                part = MIMEBase('application', 'octet-stream')
                part.set_payload(attachment.read())

            encoders.encode_base64(part)
            filename = os.path.basename(tex_path)
            part.add_header(
                'Content-Disposition',
                f'attachment; filename={filename}'
            )
            msg.attach(part)
            logger.info(f"Fichier TEX joint: {filename}")

        # Envoi de l'email
        try:
            logger.info("Connexion au serveur SMTP Gmail...")
            server = smtplib.SMTP('smtp.gmail.com', 587)
            server.starttls()
            server.login(self.gmail_user, self.gmail_password)

            # Liste complète des destinataires
            all_recipients = to_emails + (cc_emails or [])

            text = msg.as_string()
            server.sendmail(self.gmail_user, all_recipients, text)
            server.quit()

            logger.info(f"Email envoyé avec succès à: {', '.join(to_emails)}")

        except smtplib.SMTPAuthenticationError:
            logger.error("Erreur d'authentification Gmail. Vérifiez vos identifiants.")
            raise
        except smtplib.SMTPException as e:
            logger.error(f"Erreur SMTP: {e}")
            raise
        except Exception as e:
            logger.error(f"Erreur lors de l'envoi de l'email: {e}")
            raise


def SendReport(
        GMAIL_USER_,
        GMAIL_APP_PASSWORD_,
        LATEX_FILE_,
        TO_EMAILS_,
        CC_EMAILS_,
        SUBJECT_,
        EMAIL_BODY_
):
    """Fonction principale - exemple d'utilisation"""

    # ================================
    # CONFIGURATION - À MODIFIER
    # ================================

    # Identifiants Gmail
    GMAIL_USER = GMAIL_USER_
    GMAIL_APP_PASSWORD = GMAIL_APP_PASSWORD_  # Pas votre mot de passe normal !

    # Fichier LaTeX à compiler
    LATEX_FILE = LATEX_FILE_  # Chemin vers votre fichier .tex

    # Configuration email
    TO_EMAILS = TO_EMAILS_
    CC_EMAILS = CC_EMAILS_
    SUBJECT = SUBJECT_

    EMAIL_BODY = EMAIL_BODY_


    # ================================
    # EXÉCUTION
    # ================================

    try:
        # Initialisation
        sender = LaTeXEmailSender(GMAIL_USER, GMAIL_APP_PASSWORD)

        # Compilation LaTeX
        logger.info("=== DÉBUT COMPILATION LATEX ===")
        pdf_path = sender.compile_latex(LATEX_FILE)
        logger.info("=== FIN COMPILATION LATEX ===")

        # Envoi par email
        logger.info("=== DÉBUT ENVOI EMAIL ===")
        sender.send_email(
            to_emails=TO_EMAILS,
            cc_emails=CC_EMAILS,
            subject=SUBJECT,
            body=EMAIL_BODY,
            pdf_path=pdf_path,
            tex_path=LATEX_FILE
        )
        logger.info("=== FIN ENVOI EMAIL ===")

        logger.info("🎉 Processus terminé avec succès!")

    except FileNotFoundError as e:
        logger.error(f"❌ Fichier non trouvé: {e}")
        sys.exit(1)
    except RuntimeError as e:
        logger.error(f"❌ Erreur de compilation: {e}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"❌ Erreur inattendue: {e}")
        sys.exit(1)
