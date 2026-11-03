# Odoo 17 Inventory Module - Warehouse Operations Guide

Owner: Supply Chain Systems team. Applies to the Northwind Distribution Odoo 17 instance.

## Warehouses and routes

We run two warehouses: WH-DHK (Dhaka, main hub) and WH-CTG (Chittagong, port warehouse).
WH-DHK uses 2-step receipts (Input -> Stock) so every inbound shipment passes a quality
check in the Input location before it becomes available. WH-CTG uses 1-step receipts.

Outgoing deliveries from WH-DHK use 3-step delivery: Pick -> Pack -> Ship. WH-CTG uses
2-step delivery (Pick -> Ship) because it only ships full pallets.

## Removal strategies

The default removal strategy is FIFO (first in, first out). Perishable products in the
"Food & Beverage" category use FEFO (first expired, first out), which requires the
expiration date to be set on every lot. Lot tracking is mandatory for the "Pharma" category.

## Reordering rules

Reordering rules for the "Consumables" category use a minimum of 50 units and a maximum of
200 units. When on-hand plus incoming quantity drops below the minimum, Odoo creates a
request for quotation to bring the stock back up to the maximum.

The replenishment scheduler runs daily at 02:00 (Asia/Dhaka). The default vendor lead
time is 7 days and the purchase security lead time is 2 days, so Odoo orders 9 days
before the forecasted shortage.

## Inventory adjustments and counting

Any inventory adjustment with a variance above 5% or a value above 10,000 BDT must be
approved by the Inventory Manager before it is applied. Smaller adjustments can be
validated by a Warehouse Supervisor.

Cycle counts follow ABC classification: A items are counted monthly, B items quarterly,
and C items once a year.

## Barcodes

Scanners use the GS1 barcode nomenclature. Lot numbers are printed on GS1-128 labels.
