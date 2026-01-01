import { PODocument } from "@/lib/types";
import {
    Table,
    TableBody,
    TableCell,
    TableHead,
    TableHeader,
    TableRow,
} from "@/components/ui/table";
import { Input } from "@/components/ui/input";
import { Button } from "@/components/ui/button";
import { useState } from "react";
import { ExternalLink, ArrowUpDown, ArrowUp, ArrowDown, RotateCw } from "lucide-react";
import { formatNumber, formatQuantity } from "@/lib/utils";

interface HistoryTableProps {
    data: PODocument[];
    onRefresh?: () => void;
    isRefreshing?: boolean;
}

type SortDirection = 'asc' | 'desc';

export function HistoryTable({ data, onRefresh, isRefreshing = false }: HistoryTableProps) {
    const [searchTerm, setSearchTerm] = useState("");
    const [currentPage, setCurrentPage] = useState(1);
    const [sortColumn, setSortColumn] = useState<keyof PODocument | null>(null);
    const [sortDirection, setSortDirection] = useState<SortDirection>('asc');

    const itemsPerPage = 20;

    // Filter
    const filteredData = data.filter((doc) => {
        const searchLower = searchTerm.toLowerCase();
        return (
            (doc.po_number?.toLowerCase() || "").includes(searchLower) ||
            (doc.retailer_name?.toLowerCase() || "").includes(searchLower) ||
            (doc.branch_name?.toLowerCase() || "").includes(searchLower) ||
            (doc.debtor_code?.toLowerCase() || "").includes(searchLower)
        );
    });

    // Sort
    const sortedData = [...filteredData].sort((a, b) => {
        if (!sortColumn) return 0;

        let valA = a[sortColumn];
        let valB = b[sortColumn];

        // Handle specific nested or numeric fields if needed for sorting
        // For now, doing simple string/number comparison
        if (typeof valA === 'string') valA = valA.toLowerCase();
        if (typeof valB === 'string') valB = valB.toLowerCase();

        if (valA === valB) return 0;

        const comparison = valA! > valB! ? 1 : -1;
        return sortDirection === 'asc' ? comparison : -comparison;
    });

    const totalItems = sortedData.length;
    const totalPages = Math.ceil(totalItems / itemsPerPage);
    const startIndex = (currentPage - 1) * itemsPerPage;
    const endIndex = Math.min(startIndex + itemsPerPage, totalItems);
    const currentData = sortedData.slice(startIndex, startIndex + itemsPerPage);

    const handleSort = (column: keyof PODocument) => {
        if (sortColumn === column) {
            setSortDirection(sortDirection === 'asc' ? 'desc' : 'asc');
        } else {
            setSortColumn(column);
            setSortDirection('asc');
        }
    };

    const SortIcon = ({ column }: { column: keyof PODocument }) => {
        if (sortColumn !== column) return <ArrowUpDown className="ml-2 h-4 w-4 opacity-50" />;
        return sortDirection === 'asc' ? <ArrowUp className="ml-2 h-4 w-4" /> : <ArrowDown className="ml-2 h-4 w-4" />;
    };

    return (
        <div className="space-y-4">
            <div className="flex flex-col md:flex-row justify-between items-center gap-4">
                <div className="flex items-center gap-2 w-full md:w-auto">
                    <Input
                        placeholder="Search by PO Number, Retailer, Debtor Code, or Branch"
                        value={searchTerm}
                        onChange={(e) => {
                            setSearchTerm(e.target.value);
                            setCurrentPage(1);
                        }}
                        className="w-full md:min-w-[400px]"
                    />
                    <Button
                        variant="outline"
                        onClick={onRefresh}
                        disabled={isRefreshing}
                        title="Refresh"
                        className="gap-2"
                    >
                        <RotateCw className={`h-4 w-4 ${isRefreshing ? 'animate-spin' : ''}`} />
                        Refresh Data
                    </Button>
                </div>

                <div className="text-sm text-muted-foreground whitespace-nowrap">
                    {totalItems > 0 ? (
                        <span>Showing {startIndex + 1} to {endIndex} of {totalItems} entries</span>
                    ) : (
                        <span>No entries found</span>
                    )}
                </div>
            </div>

            <div className="rounded-md border overflow-x-auto">
                <Table>
                    <TableHeader>
                        <TableRow>
                            <TableHead className="min-w-[120px] cursor-pointer hover:bg-muted/50 transition-colors" onClick={() => handleSort('po_number')}>
                                <div className="flex items-center">PO Number <SortIcon column="po_number" /></div>
                            </TableHead>
                            <TableHead className="min-w-[150px] cursor-pointer hover:bg-muted/50" onClick={() => handleSort('retailer_name')}>
                                <div className="flex items-center">Retailer <SortIcon column="retailer_name" /></div>
                            </TableHead>
                            <TableHead className="min-w-[100px] cursor-pointer hover:bg-muted/50" onClick={() => handleSort('debtor_code')}>
                                <div className="flex items-center">Debtor <SortIcon column="debtor_code" /></div>
                            </TableHead>
                            <TableHead className="min-w-[150px] cursor-pointer hover:bg-muted/50" onClick={() => handleSort('branch_name')}>
                                <div className="flex items-center">Branch <SortIcon column="branch_name" /></div>
                            </TableHead>
                            <TableHead className="min-w-[100px] cursor-pointer hover:bg-muted/50" onClick={() => handleSort('branch_code')}>
                                <div className="flex items-center">Branch Code <SortIcon column="branch_code" /></div>
                            </TableHead>
                            <TableHead className="min-w-[150px] cursor-pointer hover:bg-muted/50" onClick={() => handleSort('buyer_name')}>
                                <div className="flex items-center">Buyer Name <SortIcon column="buyer_name" /></div>
                            </TableHead>
                            <TableHead className="min-w-[180px] cursor-pointer hover:bg-muted/50" onClick={() => handleSort('delivery_address')}>
                                <div className="flex items-center">Delivery Address <SortIcon column="delivery_address" /></div>
                            </TableHead>
                            <TableHead className="min-w-[100px] cursor-pointer hover:bg-muted/50" onClick={() => handleSort('po_date')}>
                                <div className="flex items-center">PO Date <SortIcon column="po_date" /></div>
                            </TableHead>
                            <TableHead className="min-w-[120px] cursor-pointer hover:bg-muted/50" onClick={() => handleSort('delivery_date')}>
                                <div className="flex items-center">Delivery Date <SortIcon column="delivery_date" /></div>
                            </TableHead>
                            <TableHead className="min-w-[110px] cursor-pointer hover:bg-muted/50" onClick={() => handleSort('expiry_date')}>
                                <div className="flex items-center">Expiry Date <SortIcon column="expiry_date" /></div>
                            </TableHead>
                            <TableHead className="min-w-[70px] cursor-pointer hover:bg-muted/50" onClick={() => handleSort('currency')}>
                                <div className="flex items-center">Currency <SortIcon column="currency" /></div>
                            </TableHead>
                            <TableHead className="min-w-[100px] cursor-pointer hover:bg-muted/50" onClick={() => handleSort('total_amount')}>
                                <div className="flex items-center">Total <SortIcon column="total_amount" /></div>
                            </TableHead>
                            <TableHead className="min-w-[120px]">Tax ID</TableHead>
                            <TableHead className="min-w-[110px]">Article Code</TableHead>
                            <TableHead className="min-w-[130px]">Barcode</TableHead>
                            <TableHead className="min-w-[200px]">Description</TableHead>
                            <TableHead className="min-w-[70px] cursor-pointer hover:bg-muted/50" onClick={() => handleSort('qty' as keyof PODocument)}>
                                <div className="flex items-center">Qty <SortIcon column={'qty' as keyof PODocument} /></div>
                            </TableHead>
                            <TableHead className="min-w-[70px] cursor-pointer hover:bg-muted/50" onClick={() => handleSort('uom' as keyof PODocument)}>
                                <div className="flex items-center">UOM <SortIcon column={'uom' as keyof PODocument} /></div>
                            </TableHead>
                            <TableHead className="min-w-[90px] cursor-pointer hover:bg-muted/50" onClick={() => handleSort('unit_price' as keyof PODocument)}>
                                <div className="flex items-center">Unit Price <SortIcon column={'unit_price' as keyof PODocument} /></div>
                            </TableHead>
                            <TableHead className="min-w-[90px] cursor-pointer hover:bg-muted/50" onClick={() => handleSort('line_total' as keyof PODocument)}>
                                <div className="flex items-center">Line Total <SortIcon column={'line_total' as keyof PODocument} /></div>
                            </TableHead>
                            <TableHead className="min-w-[140px]">Source File</TableHead>
                        </TableRow>
                    </TableHeader>
                    <TableBody>
                        {currentData.length > 0 ? (
                            currentData.map((doc, idx) => (
                                <TableRow key={idx}>
                                    <TableCell className="font-medium">{doc.po_number || "-"}</TableCell>
                                    <TableCell>{doc.retailer_name || "-"}</TableCell>
                                    <TableCell>{doc.debtor_code || "-"}</TableCell>
                                    <TableCell>{doc.branch_name || "-"}</TableCell>
                                    <TableCell>{doc.branch_code || "-"}</TableCell>
                                    <TableCell>{doc.buyer_name || "-"}</TableCell>
                                    <TableCell className="max-w-[180px] truncate" title={doc.delivery_address}>
                                        {doc.delivery_address || "-"}
                                    </TableCell>
                                    <TableCell>{doc.po_date || "-"}</TableCell>
                                    <TableCell>{doc.delivery_date || "-"}</TableCell>
                                    <TableCell>{doc.expiry_date || "-"}</TableCell>
                                    <TableCell>{doc.currency || "-"}</TableCell>
                                    <TableCell>{formatNumber(doc.total_amount)}</TableCell>
                                    <TableCell>{doc.tax_id || "-"}</TableCell>
                                    <TableCell>{(doc as any).article_code || "-"}</TableCell>
                                    <TableCell>{(doc as any).barcode || "-"}</TableCell>
                                    <TableCell className="max-w-[200px] truncate" title={(doc as any).article_description}>
                                        {(doc as any).article_description || "-"}
                                    </TableCell>
                                    <TableCell>{formatQuantity((doc as any).qty)}</TableCell>
                                    <TableCell>{(doc as any).uom || "-"}</TableCell>
                                    <TableCell>{formatNumber((doc as any).unit_price)}</TableCell>
                                    <TableCell>{formatNumber((doc as any).line_total)}</TableCell>
                                    <TableCell>
                                        {doc.file_path_url || doc.file_storage_url ? (
                                            <a href={doc.file_path_url || doc.file_storage_url} target="_blank" rel="noopener noreferrer">
                                                <Button variant="ghost" size="sm" className="hover:underline cursor-pointer">
                                                    <ExternalLink className="h-3 w-3 mr-1" />
                                                    {doc.source_filename || "View"}
                                                </Button>
                                            </a>
                                        ) : "-"}
                                    </TableCell>
                                </TableRow>
                            ))
                        ) : (
                            <TableRow>
                                <TableCell colSpan={21} className="text-left h-24 text-muted-foreground">
                                    No records found.
                                </TableCell>
                            </TableRow>
                        )}
                    </TableBody>
                </Table>
            </div>

            <div className="text-xs text-muted-foreground">
                💡 Tip: Click headers to sort
            </div>

            {totalPages > 1 && (
                <div className="flex items-center justify-end space-x-2 py-4">
                    <Button
                        variant="outline"
                        size="sm"
                        onClick={() => setCurrentPage((p) => Math.max(1, p - 1))}
                        disabled={currentPage === 1}
                    >
                        Previous
                    </Button>
                    <div className="text-sm">
                        Page {currentPage} of {totalPages}
                    </div>
                    <Button
                        variant="outline"
                        size="sm"
                        onClick={() => setCurrentPage((p) => Math.min(totalPages, p + 1))}
                        disabled={currentPage === totalPages}
                    >
                        Next
                    </Button>
                </div>
            )}
        </div>
    );
}
