"""
SeedSense AI - Professional PDF Report Generator (ReportLab)
Generates downloadable PDF analysis reports containing summary statistics,
annotated images, seed type distribution, and prediction table.
"""

import os
from datetime import datetime
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.units import inch
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image as RLImage, HRFlowable
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle

from backend.config import RESULTS_DIR

def generate_pdf_report(analysis_data, output_path):
    """
    Generates PDF report using ReportLab.
    """
    doc = SimpleDocTemplate(
        output_path,
        pagesize=letter,
        rightMargin=36,
        leftMargin=36,
        topMargin=36,
        bottomMargin=36
    )
    
    styles = getSampleStyleSheet()
    
    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Heading1'],
        fontName='Helvetica-Bold',
        fontSize=22,
        textColor=colors.HexColor('#1E293B'),
        spaceAfter=6,
        alignment=1 # Center
    )
    
    subtitle_style = ParagraphStyle(
        'DocSubTitle',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=11,
        textColor=colors.HexColor('#64748B'),
        spaceAfter=15,
        alignment=1
    )
    
    heading2_style = ParagraphStyle(
        'DocHeading2',
        parent=styles['Heading2'],
        fontName='Helvetica-Bold',
        fontSize=14,
        textColor=colors.HexColor('#0F172A'),
        spaceBefore=12,
        spaceAfter=6
    )
    
    normal_style = styles['Normal']
    normal_style.fontSize = 10
    
    elements = []
    
    # Title & Header
    elements.append(Paragraph("SeedSense AI", title_style))
    elements.append(Paragraph("Multi-Seed Identification & Quality Defect Inspection Report", subtitle_style))
    elements.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor('#2563EB'), spaceAfter=15))
    
    # Report Metadata Table
    date_str = analysis_data.get('date', datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    meta_data = [
        [Paragraph("<b>Report ID:</b>", normal_style), Paragraph(f"#{analysis_data.get('id', 'N/A')}", normal_style),
         Paragraph("<b>Date & Time:</b>", normal_style), Paragraph(date_str, normal_style)],
        [Paragraph("<b>Original File:</b>", normal_style), Paragraph(analysis_data.get('filename', 'seed_image.jpg'), normal_style),
         Paragraph("<b>Model Used:</b>", normal_style), Paragraph(f"MobileNetV3 ({analysis_data.get('model_used', 'DL')})", normal_style)]
    ]
    meta_table = Table(meta_data, colWidths=[1.2*inch, 2.3*inch, 1.2*inch, 2.3*inch])
    meta_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), colors.HexColor('#F8FAFC')),
        ('BOX', (0,0), (-1,-1), 1, colors.HexColor('#E2E8F0')),
        ('INNERGRID', (0,0), (-1,-1), 0.5, colors.HexColor('#E2E8F0')),
        ('PADDING', (0,0), (-1,-1), 6),
    ]))
    elements.append(meta_table)
    elements.append(Spacer(1, 15))
    
    # Summary Cards Table
    tot_seeds = analysis_data.get('total_seeds', 1)
    healthy_cnt = analysis_data.get('healthy_count', 0)
    defective_cnt = analysis_data.get('defective_count', 0)
    healthy_pct = analysis_data.get('healthy_percentage', 0.0)
    defective_pct = 100.0 - healthy_pct if tot_seeds > 0 else 0.0
    
    summary_data = [
        [Paragraph("Total Seeds", normal_style), Paragraph("Healthy Seeds", normal_style), Paragraph("Defective Seeds", normal_style), Paragraph("Healthy %", normal_style)],
        [Paragraph(f"<b>{tot_seeds}</b>", title_style), Paragraph(f"<b>{healthy_cnt}</b>", title_style), Paragraph(f"<b>{defective_cnt}</b>", title_style), Paragraph(f"<b>{healthy_pct:.1f}%</b>", title_style)]
    ]
    summary_table = Table(summary_data, colWidths=[1.75*inch, 1.75*inch, 1.75*inch, 1.75*inch])
    summary_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), colors.HexColor('#EFF6FF')),
        ('ALIGN', (0,0), (-1,-1), 'CENTER'),
        ('BOX', (0,0), (-1,-1), 1, colors.HexColor('#BFDBFE')),
        ('PADDING', (0,0), (-1,-1), 8),
    ]))
    elements.append(summary_table)
    elements.append(Spacer(1, 15))
    
    # Annotated Image
    elements.append(Paragraph("Visual Seed Segmentation & Annotation", heading2_style))
    annotated_filename = analysis_data.get('annotated_filename', 'annotated_result.jpg')
    annotated_path = os.path.join(RESULTS_DIR, annotated_filename)
    
    if os.path.exists(annotated_path):
        try:
            img = RLImage(annotated_path, width=5.5*inch, height=3.5*inch)
            elements.append(img)
        except Exception as e:
            elements.append(Paragraph(f"[Image Render Warning: {e}]", normal_style))
    else:
        elements.append(Paragraph("[Annotated image file not found]", normal_style))
        
    elements.append(Spacer(1, 15))
    
    # Detailed Seed Predictions Table
    elements.append(Paragraph("Individual Seed Quality & Defect Table", heading2_style))
    
    table_data = [
        [Paragraph("<b>#</b>", normal_style), Paragraph("<b>Condition</b>", normal_style), Paragraph("<b>Defect Type</b>", normal_style), Paragraph("<b>Confidence</b>", normal_style)]
    ]
    
    seeds = analysis_data.get('seeds', [])
    if not seeds and 'primary_condition' in analysis_data:
        seeds = [{
            'seed_index': 1,
            'condition': analysis_data.get('primary_condition'),
            'defect_type': analysis_data.get('primary_defect_type'),
            'confidence': analysis_data.get('primary_confidence')
        }]
        
    for s in seeds[:30]: # Cap at 30 items for PDF rendering
        idx = s.get('seed_index', 1)
        cond = s.get('condition', 'Healthy')
        defect = s.get('defect_type', 'None')
        conf = f"{float(s.get('confidence', 0.0))*100:.1f}%"
        
        table_data.append([
            Paragraph(str(idx), normal_style),
            Paragraph(f"<font color='{'#16A34A' if 'Healthy' in cond else '#DC2626'}'><b>{cond}</b></font>", normal_style),
            Paragraph(defect, normal_style),
            Paragraph(conf, normal_style)
        ])
        
    pred_table = Table(table_data, colWidths=[0.8*inch, 2.3*inch, 2.3*inch, 1.6*inch])
    pred_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#F1F5F9')),
        ('BOX', (0,0), (-1,-1), 1, colors.HexColor('#CBD5E1')),
        ('INNERGRID', (0,0), (-1,-1), 0.5, colors.HexColor('#E2E8F0')),
        ('PADDING', (0,0), (-1,-1), 5),
    ]))
    elements.append(pred_table)
    
    # Build Document
    doc.build(elements)
    return output_path
